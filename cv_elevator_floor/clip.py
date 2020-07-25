
import cv2
import sys

def resize(inname,outname):
# 開啟影片檔案
  cap = cv2.VideoCapture(inname)

  # 取得畫面尺寸
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # 使用 XVID 編碼
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  # 建立 VideoWriter 物件，輸出影片至 output.avi，FPS 值為 20.0
  out = cv2.VideoWriter(outname, fourcc, 3, (13 , 13))

  ret = True
  # 以迴圈從影片檔案讀取影格，並顯示出來
  ct = 0
  while(ret):

    ret, frame = cap.read()

    if ret:
        #print(frame.shape)
        frame = frame[25 : 45, 1534 : 1562]
        #print(frame.shape)
        cv2.imshow('img',frame)
        if ct % 5 == 0:
        	out.write(frame)
        ct += 1
    # 顯示結果
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  out.release()
  cv2.destroyAllWindows()
if __name__ == '__main__':
  resize(sys.argv[1],sys.argv[2])