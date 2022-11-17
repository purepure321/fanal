import numpy as np
import cv2 

def preprocessing(img) : 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # รับค่า img โดยใช้ cv2.cvtColor แปลง model สีเป็น Gray *** BGR2GRAY *** เก็บค่าใน img_gray
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img, img_gray) # return ค่า (img, img_gray)

def feature_object_detection(template_img, template_gray, query_img, query_gray, min_match_number) : #ฟังก์ชันย่อย min_match_number จำนวนเเมตน้อยสุดยอมรับได้
    template_kpts, template_desc = cv2.SIFT_create().detectAndCompute(template_gray, None) # หาจุดเด่นของภาพ โดย detectAndCompute
    query_kpts, query_desc = cv2.SIFT_create().detectAndCompute(query_gray, None)# หาจุดเด่นของภาพ โดย detectAndCompute
    
    matches = cv2.BFMatcher().knnMatch(template_desc, query_desc, k=2) # templat_desc กับ query_desc มา matches
    good_matches = list() #เก็บค่า
    good_matches_list = list() #เก็บค่า

    for m, n in matches : # matches[0],matches[1]
        if m.distance < 0.7*n.distance : #ถ้าระยะ m น้อยกว่า n เป็น good matches ที่ดี
            good_matches.append(m)
            good_matches_list.append([m])
    
    if len(good_matches) > min_match_number : # len นับ good_matches > min_match_number 
        src_pts = np.float32([ template_kpts[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2) #วน for ตาม จน good match ได้ src_pts
        dst_pts = np.float32([ query_kpts[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2) #วน for ตาม จน good match ได้ dst_pts

        H, inlier_masks = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0) # H RANSAC
        # get the bounding box around template image
        h, w = template_img.shape[:2]
        template_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2) # -1 ข้อมูลเป้นรก็ได้ 1เเถว 2หลัก
        transformed_box = cv2.perspectiveTransform(template_box, H)

        detected_img = cv2.polylines(query_img, [np.int32(transformed_box)], True, (0,0,255), 3, cv2.LINE_AA)# plot line
        drawmatch_img = cv2.drawMatchesKnn(template_img, template_kpts, detected_img, query_kpts, good_matches_list, None, flags=2, matchesMask=inlier_masks)

        return detected_img, drawmatch_img 
    else :
        print('Keypoints not enough')
        return 
template_img = cv2.imread('Template2.png')# อ่านค่า template_img จากโฟลเดอร์
template_img, template_gray = preprocessing(template_img) # รับค่่าจากบรรทัดก่อนหน้า ทำ preprocessing เก็บค่า template_img, template_gray 

query_vdo1 = cv2.VideoCapture('right_output.avi')
#query_vdo1, template_gray = preprocessing(query_vdo1)

while query_vdo1.isOpened() :
    ret, frame = query_vdo1.read() #ret=true
    if ret : # ถ้าอ่านสำเร็จส่งค่า frame
        query_vdo_gray1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        detected, drawmatch =  feature_object_detection(template_img, template_gray, frame, query_vdo_gray1, 1) #๑# เรียกฟังก์ชันย่อย feature_object_detection อ่านค่ารับค่าจาก 4 ตัวในวงเล็บ ฟังก์ชันย่อย feature_object_detection ส่งค่า detected, drawmatch
        cv2.imshow('Video', frame)# เเสดง Video frame รับค่าจาก  frame                                    

        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q') : # this line control the period between image frame
            break         #1000/30 ค่า delay vdo 30 ms
    else :
        break
query_vdo1.release()
cv2.destroyAllWindows()#กดออกเเล้วออกทั้งหมด