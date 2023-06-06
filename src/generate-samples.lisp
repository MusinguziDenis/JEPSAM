(in-package :pp-cust)

(defparameter *move-command-fraction* 0.333)
(defparameter *corner-1* '(-2.6 -1.1)) ; -2.7, -1.1
(defparameter *corner-2* '(-2.1 -1.2)) ; -2.1, -1.3
(defparameter *corner-3* '(-1.9 -1.1)) ; -1.7, -1.1
(defparameter *pickable-objects* '(:bottle :cup))
(defparameter *placeable-objects* '(:SPOON :RED-METAL-PLATE :MILK :CUP :CUBE :CEREAL :BUTTERMILK :BREAKFAST-CEREAL :BOTTLE :BLUE-METAL-PLATE :MUG :PLATE :MONDAMIN :POT :WEISSWURST :BOWL :FORK :KNIFE :SPATULA :CAP :GLASSES :GLOVE :SHOE))
(defparameter *available-colors* '(red blue green))
(defparameter *available-poses* nil)
(defparameter *obj2-poses* '(pose-1 pose-2 pose-3 pose-4 pose-5))
(defparameter *obj1-poses* '(pose-6 pose-7 pose-8 pose-9 pose-10))

(defparameter *vector-1* (list (- (first *corner-1*) (first *corner-3*))
                               (- (second *corner-1*) (second *corner-3*))))
(defparameter *put-transforms* (list #'*leftward-transformation* #'*rightward-transformation* #'*backward-transformation* #'*forward-transformation*))
(defparameter *move-transforms* (list #'*leftward-transformation* #'*rightward-transformation* #'*backward-transformation* #'*forward-transformation*))

(defparameter *vector-2* (list (- (first *corner-2*) (first *corner-3*))
                               (- (second *corner-2*) (second *corner-3*))))
(defparameter *v1* nil)
(defparameter *v2* nil)

(defparameter *u1* 0.0)
(defparameter *u2* 0.0)

(defparameter *obj1-location* nil)
(defparameter *obj2-location* nil)
(defparameter *obj1-type* nil)
(defparameter *obj2-type* nil)
(defparameter *obj1-color* nil)
(defparameter *obj2-color* nil)
(defparameter *transform* nil)

(defun add-vectors (x y)
  (list (+ (first x) (first y))
        (+ (second x) (second y))))

(defun random-choice (x)
  (let* ((length-list (length x))
         (random-idx (random length-list)))
    (nth random-idx x)))

(defun setup-generation ()
  (setf (gethash "left" *actions-hash*) #'*leftward-transformation*)
  (setf (gethash "right" *actions-hash*) #'*rightward-transformation*)
  (setf (gethash "forward" *actions-hash*) #'*forward-transformation*)
  (setf (gethash "backward" *actions-hash*) #'*backward-transformation*)


  (setf (gethash 'pose-1 *pose-hash*) *object-location-1*)
  (setf (gethash 'pose-2 *pose-hash*) *object-location-2*)
  (setf (gethash 'pose-3 *pose-hash*) *object-location-3*)
  (setf (gethash 'pose-4 *pose-hash*) *object-location-4*)
  (setf (gethash 'pose-5 *pose-hash*) *object-location-5*)
  (setf (gethash 'pose-6 *pose-hash*) *object-location-6*)
  (setf (gethash 'pose-7 *pose-hash*) *object-location-7*)
  (setf (gethash 'pose-8 *pose-hash*) *object-location-8*)
  (setf (gethash 'pose-9 *pose-hash*) *object-location-9*)
  (setf (gethash 'pose-10 *pose-hash*) *object-location-10*)
  
 ;; (dotimes (n 10)
 ;;   (setf (gethash (read-from-string (concatenate 'string "pose-" (write-to-string (+ 15 n)))) *pose-hash*)
 ;;                  (make-pose "map" (generate-random-location))))

  (setf (gethash #'*leftward-transformation* *reposition-hash*) "left ")
  (setf (gethash #'*rightward-transformation* *reposition-hash*) "right ")
  (setf (gethash #'*forward-transformation* *reposition-hash*) "forwards ")
  (setf (gethash #'*backward-transformation* *reposition-hash*) "backwards ")

  (setf (gethash  #'*leftward-transformation* *transport-hash*) "to the left of ")
  (setf (gethash #'*rightward-transformation* *transport-hash*) "to the right of ")
  (setf (gethash #'*forward-transformation* *transport-hash*) "in front of ")
  (setf (gethash #'*backward-transformation* *transport-hash*) "behind ")
  (setf (gethash #'*on-transformation* *transport-hash*) "on top of ")

  (setf (gethash #'*leftward-transformation* *transformation-hash*) " #'*leftward-transformation* ")
  (setf (gethash #'*rightward-transformation* *transformation-hash*) " #'*rightward-transformation* ")
  (setf (gethash #'*forward-transformation* *transformation-hash*) " #'*forward-transformation* ")
  (setf (gethash #'*backward-transformation* *transformation-hash*) " #'*backward-transformation* ")

  (setf (gethash 'red *color-hash*) '(1 0 0))
  (setf (gethash 'green *color-hash*) '(0 1 0))
  (setf (gethash 'blue *color-hash*) '(0 0 1))

  (setf *available-poses* (alexandria:hash-table-keys *pose-hash*))
)

(defun generate-random-location ()
  (setf *u1* (random 1.0))
  (setf *u2* (random 1.0))
  (if (> (+ *u1* *u2*) 1.0)
      (progn
        (setf *u1* (- 1 *u1*))
        (setf *u2* (- 1 *u2*))))
  
  (setf *v1* (mapcar (lambda ( x ) (* x *u1*)) *vector-1*))
  (setf *v2* (mapcar (lambda ( x ) (* x *u2*)) *vector-2*))
  (setf *v2* (add-vectors *v1* *v2*))
  (setf *v2* (add-vectors *corner-3* *v2*))
  (list (list (first *v2*) (second *v2*) 0.82) (list 0 0 0 1)))

(defun generate-dataset (num-samples &optional (start-id 0))
  (setup-generation)
  (ignore-errors
  (dotimes (counter num-samples)
    (init-projection)
    (generate-random-sample (+ counter start-id))
    (sleep 0.5)))
  (init-projection)
)

(defun generate-random-sample (sim-id)
  ;;(init-projection)
  (setf *simulation-id* (write-to-string sim-id))
  
  (if (> (random 1.0) *move-command-fraction*)
      (progn
        (setf *obj1-location* (random-choice *obj1-poses*))
        (setf *obj2-location* (random-choice *obj2-poses*))
        (setf *transform* (random-choice *put-transforms*))
        (setf *obj1-type* (random-choice *pickable-objects*))
        (setf *obj2-type* (random-choice *placeable-objects*))
        (setf *obj1-color* (random-choice *available-colors*))
        (setf *obj2-color* (random-choice *available-colors*))
        (motor-program (list (list *obj1-type* *obj1-color* *obj1-location*)
                             (list *obj2-type* *obj2-color* *obj2-location*))
                       *obj1-type* *transform* *obj2-type*))
      (progn
        (setf *obj1-location* (random-choice *obj2-poses*))
        (setf *transform* (random-choice *move-transforms*))
        (setf *obj1-type* (random-choice *pickable-objects*))
        (setf *obj1-color* (random-choice *available-colors*))
        (motor-program (list (list *obj1-type* *obj1-color* *obj1-location*))
                       *obj1-type* *transform* *obj1-type*))))
  
(defun call-me-maybe()
  (init-projection)
(setf *simulation-id* "0")
(motor-program '((:cup blue pose-1) (:bowl green pose-2)) :cup #'*leftward-transformation* :bowl)
(sleep 0.5)
(init-projection)

(setf *simulation-id* "1")
(motor-program '((:bottle blue pose-2) (:cup green pose-3)) :bottle #'*forward-transformation* :cup)
(sleep 0.5)
(init-projection)

(setf *simulation-id* "2")
(motor-program '((:bottle red pose-3) (:mug red pose-2)) :bottle #'*backward-transformation* :mug)
(sleep 0.5)
(init-projection)

(setf *simulation-id* "3")
(motor-program '((:bottle blue pose-1) (:cup red pose-2)) :bottle #'*rightward-transformation* :cup)
(sleep 0.5)
(init-projection)

(setf *simulation-id* "4")
(motor-program '((:cup blue pose-1)) :cup #'*leftward-transformation* :cup)
(sleep 0.5)
(init-projection))

