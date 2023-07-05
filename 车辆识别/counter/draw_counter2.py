import cv2
def draw_up_down_counter(img, up_counter, down_counter, frame_feature, names):
    font_size = 2
    font_cuxi = 3
    top_left = frame_feature[1]*4//5
    cv2.rectangle(img, (0, top_left), (frame_feature[0]//4, frame_feature[1]), (237,149,100), thickness=-1)
    cv2.putText(img, 'veh_type', (10, 40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), font_cuxi)
    text_size = cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, thickness=-1)
    cv2.putText(img, 'up', (250, 40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), font_cuxi)
    cv2.putText(img, 'down', (350, 40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), font_cuxi)
    for i, name in enumerate(names):
        cv2.putText(img, '%s' %name, (10, (i+2)*40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0,0,0),font_cuxi)
        cv2.putText(img, '%s' %str(up_counter[i]), ((int(text_size[0][0]) + 30), (i+2)*40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0),font_cuxi)
        cv2.putText(img, '%s' %str(down_counter[i]), ((int(text_size[0][0]) + 120), (i+2)*40+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), font_cuxi)
    cv2.putText(img,'Total',(10,200+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0, 0, 0), font_cuxi)
    cv2.putText(img, '%s' %str(sum(up_counter)),(int(text_size[0][0]) + 30,200+top_left),cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, (0,0,0),font_cuxi)
    cv2.putText(img, '%s' % str(sum(down_counter)), (int(text_size[0][0]) + 120,200+top_left), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,(0, 0, 0), font_cuxi)
