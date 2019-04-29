import cv2
import dlib
#运用现成的脸部关键点检测的库
detector = dlib.get_frontal_face_detector() #使用默认的人类识别器模型
predictor = dlib.shape_predictor(
    "D:\pychar_projects\dnc-master\shape_predictor_68_face_landmarks.dat"
)
def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (0, 0, 255), 3)
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top-50), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("image", img)
cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()
    discern(img)
    if cv2.waitKey(30)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()