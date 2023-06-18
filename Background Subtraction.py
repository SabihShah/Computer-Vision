import cv2 as cv

cap = cv.VideoCapture(0)

ret, background = cap.read()
background_gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
cv.imshow("", background_gray)

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    diff = cv.absdiff(background_gray, gray)
    
    # _, foreground = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

    cv.imshow("LIVE VIDEO", diff)
    # cv.imshow("Foreground", foreground)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
