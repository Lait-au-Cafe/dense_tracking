import cv2

cap = cv2.VideoCapture(2)

while True:
   key = cv2.waitKey(10)
   if key == 27: break
   
   if key == ord('p'):
       cnt = 0
       while True:
           key = cv2.waitKey(10)
           if key == ord(' '): break
           ret, frame = cap.read()
           print(f"Export ./data/frame{cnt}.png")
           cv2.imwrite(f"./data/frame{cnt}.png", frame)
           cv2.imshow("Capture", frame)
           cnt += 1

   ret, frame = cap.read()
   cv2.imshow("Capture", frame)

cap.release()
