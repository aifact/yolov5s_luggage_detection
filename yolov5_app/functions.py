import numpy as np
import time
import cv2
import sqlite3
import api

# =============================================================================
#  Main functions
# =============================================================================


def IoL(boxA, boxB):
    """ 
    Return the Area of the intersection over the area of the smallest box
    Represent how much a box is inside an other 
    Input bbox: ( X, Y, W, H) """
        
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Area of rectangles
    boxAArea = (boxA[2]+1) * (boxA[3]+1)
    boxBArea = (boxB[2]+1) * (boxB[3]+1)
    
    iol = interArea / min(boxAArea, boxBArea)
    return(iol)
    
def ownership_decision (bbox, im0):
    for b in bbox.itertuples():
        if b.LUGGAGE ==1 and b.OWNER==0:
            for env in bbox.itertuples():
                if env.Index != b.Index and env.CLASSE=='Person':
                    shape = 2*b.WIDTH
                    iol = IoL((max(0,b.POS_X - shape), max(0,b.POS_Y-shape), b.WIDTH+2*shape, b.HEIGHT+2*shape), (env.POS_X, env.POS_Y, env.WIDTH, env.HEIGHT))
                    if iol> 0.0 :
                        bbox.loc[b.Index, 'OWNER']=1
                        owner_img = im0[env.POS_Y:(env.POS_Y+env.HEIGHT), env.POS_X:(env.POS_X+env.WIDTH)]
                        path = './images/owner_folder/owner_'+str(b.POS_X)+''+str(b.POS_Y)+'.png'
                        cv2.imwrite(path, owner_img)
                        bbox.loc[b.Index, 'OWNER_PATH']=path
        
    return(bbox)                   
                
def abandonned_decision (bbox, im0):

    for b in bbox.itertuples():
        if b.LUGGAGE ==1 and b.OWNER ==1:
            owner_img = cv2.imread(b.OWNER_PATH)
            for env in bbox.itertuples():
                if env.Index != b.Index and env.CLASSE=='Person' :
                    shape = b.WIDTH //2
                    iol = IoL((max(0,b.POS_X - shape), max(0,b.POS_Y-shape), b.WIDTH+2*shape, b.HEIGHT+2*shape), (env.POS_X, env.POS_Y, env.WIDTH, env.HEIGHT))
                    if iol > 0.0:
                        env_im = im0[env.POS_Y:(env.POS_Y+env.HEIGHT), env.POS_X:(env.POS_X+env.WIDTH)]
                        h_own, w_own = owner_img.shape[:2]
                        h_im, w_im = env_im.shape[:2]
                        if h_im<h_own :
                            h_im = h_own + 5
                        if w_im < w_own :
                            w_im = w_own + 5
                        env_im = cv2.resize(env_im, (w_im, h_im))
                        res = cv2.matchTemplate(env_im, owner_img, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        moy_val = (min_val + max_val)/2
                        print(moy_val, min_val, max_val)
                        if moy_val >=0.35 and max_val >=0.5 :
                            bbox.loc[b.Index, 'ALERT']=0
                        else :    
                            print('Not luggage owner')
                    
    return (bbox) 

def process_alert_bbox(bbox, alert_nb, image, database_path, folder_path):
        
    ind = []
    for b in bbox.itertuples():
        if b.LUGGAGE == 1 :
            if b.NEW== 1 :
                if b.ALERT==1 and b.ID_ALERT==-1 and b.NB_FRAME >=3:
                    image_path = folder_path +"/alert_image_"+str(alert_nb)+".png"
                    cv2.imwrite(image_path, image)
                    alert_nb += 1
                    idalert = write_bd(database_path, image_path, b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT, b.TIME_BEGIN)
                    bbox.loc[b.Index, 'ID_ALERT']=idalert
            else :
                if b.END == 0: 
                    if b.ALERT ==1:
                        update_bd(database_path, b.ID_ALERT, b.POS_X, b.POS_Y, b.WIDTH, b.HEIGHT)
                        image_path = folder_path +"/alert_image_"+str(alert_nb)+".png"
                        cv2.imwrite(image_path, image)
                        alert_nb += 1
                elif b.END == 1 :
                    if b.ID_ALERT != 0:
                        time_end_bd(database_path, b.ID_ALERT)
                        print('end time updating')
                    
                    bbox.loc[b.Index, 'END']=2
                elif b.END ==2 :
                    bbox.loc[b.Index, 'END']=3
                        
                elif b.END ==3 :
                    ind.append(b.Index)
        else :
            if b.END == 1 :
                ind.append(b.Index)

    # Remove ended boxes
    bbox.drop(index = ind, inplace=True)

    for i in range (bbox.shape[0]):
        bbox.NEW.values[i]=0
    
    return (bbox, alert_nb)

# =============================================================================
#  Manage time
# =============================================================================

def time_difference(time1, time2):
    """ Return the difference between 2 times """
    try:
        t1 = time.mktime(time.strptime(time1))
        t2 = time.mktime(time.strptime(time2))
        
        seconds = abs(t1-t2)
        
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        if hour == 0:
            return "%02d:%02d" % (minutes, seconds) 
        else:
            return "%d:%02d:%02d" % (hour, minutes, seconds) 
    except Exception as e:
        print("[ERREUR at time_difference] :", e)
    finally:
        return "00:00"


# =============================================================================
#  Data Base Management
# =============================================================================

def write_bd(database_path, image_path, pos_x, pos_y, width, height, time_begin):
    """ 
    Write a box in a row into the database
    ==========================================================================
    :param database_path: string containing the path of the database
    :param image_path: string containing the path of the alert image
    :param (pos_x, pos_y, width, height): bbox values
    :param time_begin: String containing the time when the luggage appears 
                    (ex:'Tue Jul 28 17:39:31 2020')        
                    
    :return id_alert: ID of the row written in the database
    :rtype: int
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
                # ID_ALERT    POS_X    POS_Y    WIDTH  HEIGHT TIME_BEGIN  TIME_END  TIME_TOTAL  IMAGE_PATH  
        alert = (cursor.lastrowid, pos_x, pos_y, width, height, time_begin, None, None, image_path)
        req = cursor.execute("INSERT INTO luggage_alert VALUES(?,?,?,?,?,?,?,?,?)", alert)
        connection.commit()
        
        req = cursor.execute("SELECT MAX(ID_ALERT) FROM luggage_alert")
        id_alert = int(cursor.fetchone()[0])
        
        api.create_alert(id_alert, pos_x, pos_y, width, height, time_begin, image_path)

    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()
        return(id_alert)

# write_bd("Output/results.db", 1, 2, 3, 4, 2)

def update_bd(database_path, id_alert, pos_x, pos_y, width, height):
    """
    Update the position of the bags in the db
    ==========================================================================
    :param database_path: string containing the path of the database
    :param id_alert: ID of the row written in the database
    :param (pos_x, pos_y, width, height): bbox values     
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        req = cursor.execute("UPDATE luggage_alert SET POS_X=?, POS_Y=?, WIDTH=?, HEIGHT=? WHERE ID_ALERT=?", 
                             (pos_x, pos_y, width, height, id_alert))
        connection.commit()

        api.update_alert(id_alert, pos_x, pos_y, width, height)

    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()
        
  
def time_end_bd(database_path, id_alert):
    """ 
    Write the final time in the db when the bag disappear
    ==========================================================================
    :param database_path: string containing the path of the database
    :param id_alert: ID of the row written in the database  
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        
        req = cursor.execute("SELECT TIME_BEGIN FROM luggage_alert WHERE ID_ALERT = ?", (str(id_alert),)) 
        time_begin = cursor.fetchone()[0]
        time_end = time.asctime()
        time_dif = time_difference(time_begin, time_end)
        
        req = cursor.execute("UPDATE luggage_alert SET TIME_END=?, TIME_TOTAL=?  WHERE ID_ALERT=?", 
                              (time_end, time_dif, id_alert))
        connection.commit()

        api.update_alert(id_alert, None, None, None, None, time_end)

    except Exception as e:
        print("[ERREUR At time_end_bd] :", e)
        connection.rollback()
    finally:
        connection.close()


# time_end_bd("Output/results.db", 32)

def print_db(database_path):
    """
    Print the elements of the database
    ==========================================================================
    :param database_path: string containing the path of the database
    """
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    req = cursor.execute("SELECT * FROM bagage_alerte")
    id_alert = cursor.fetchall()
    print(id_alert)
    connection.close()
    
def reset_db(database_path):
    """
    Reset the values of the database
    ==========================================================================
    :param database_path: string containing the path of the database
    """
    try:
        connection = sqlite3.connect(database_path)
        cursor = connection.cursor()
        req = cursor.execute("DELETE FROM bagage_alerte")
        connection.commit()
    except Exception as e:
        print("[ERREUR] :", e)
        connection.rollback()
    finally:
        connection.close()