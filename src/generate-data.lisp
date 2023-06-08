(in-package :pp-cust)

(defparameter *object-1-params* nil)
(defparameter *object-2-params* nil)
(defparameter *generated-description* nil)


(defparameter *actions-hash* (make-hash-table :test 'equal))
(defparameter *pose-hash* (make-hash-table :test 'equal))
(defparameter *transport-hash* (make-hash-table :test 'equal))
(defparameter *reposition-hash* (make-hash-table :test 'equal))
(defparameter *transformation-hash* (make-hash-table :test 'equal))
(defparameter *color-hash* (make-hash-table :test 'equal))


(defparameter *object-location-1* '((-1.9 -1.17 0.82) (0 0 0 1)))
(defparameter *object-location-2* '((-2.0 -1.19 0.82) (0 0 0 1)))
(defparameter *object-location-3* '((-2.1 -1.175 0.82) (0 0 0 1)))
(defparameter *object-location-4* '((-2.2 -1.18 0.82) (0 0 0 1)))
(defparameter *object-location-5* '((-2.3 -1.2 0.82) (0 0 0 1)))


(defparameter *object-location-6* '((-1.8 -1.2 0.82) (0 0 0 1)))
(defparameter *object-location-7* '((-2.0 -1.0 0.82) (0 0 0 1)))
(defparameter *object-location-8* '((-2.1 -1.0 0.82) (0 0 0 1)))
(defparameter *object-location-9* '((-2.2 -1.25 0.82) (0 0 0 1)))
(defparameter *object-location-10* '((-2.3 -1.0 0.82) (0 0 0 1)))


;;(defparameter *object-location-pose* (make-pose "map" *object-location*))

(defparameter b 200)

(defparameter *simulation-id* "3000")

(defparameter *origin*
  (make-pose "map" `((0.0 0.0 0.0) (0 0 -1 0))))

(defparameter *final-destination* nil)
(defparameter *action* nil)
(defparameter *description* "")

;;(defparameter ?navigation-goal nil)

(defparameter *auxillary-look-coordinate*
  (make-pose "base_footprint" '((1.15335 0.076 0.758) (0 0 0 1))))

(defparameter *left-downward-look-coordinate*

  (make-pose "base_footprint" '((0.65335 0.76 0.758) (0 0 0 1))))

(defparameter *right-downward-look-coordinate*

  (make-pose "base_footprint" '((0.65335 -0.76 0.758) (0 0 0 1))))

(defparameter *downward-look-coordinate*

  (make-pose "base_footprint" '((0.65335 0.076 0.758) (0 0 0 1))))

(defparameter *base-pose-near-table*
  (make-pose "map" '((-2 -0.5 0.0) (0.0 0.0 -0.7071 0.7071))))

(defparameter *base-pose-near-counter*

  (make-pose "map" '((-0.15 2 0) (0 0 -1 0))))

;;(defparameter *final-object-destination*
;;
;;  (make-pose "map" '((-0.8 2 0.9) (0 0 0 1))))

(defparameter *base-pose-near-table-towards-island*

  (make-pose "map" '((-1.447d0 0.150d0 0.0d0) (0.0d0 0.0d0 0.7071d0 0.7071d0))))

(defparameter *base-pose-near-sink-surface* 

  (make-pose "map" '((0.700000 0.650000 0.00000) (0.00000 0.00000 0 1))))

(defparameter *lift-z-offset* 0.15 "in meters")

(defparameter *lift-offset* `(0.0 0.0 ,*lift-z-offset*))

(defparameter *bottle-pregrasp-xy-offset* 0.15 "in meters")

(defparameter *bottle-grasp-xy-offset* 0.02 "in meters")

(defparameter *bottle-grasp-z-offset* 0.005 "in meters")


(defun find-auxillary-object (objects main-object-type)
  (if (equal (first (first objects)) main-object-type)
      (second objects)
      (first objects)))

(defun find-auxillary-object-type (objects main-object-type)
  (first (find-auxillary-object objects main-object-type)))

(defun find-auxillary-object-name (objects main-object-type)
  (if (equal (first (first objects)) main-object-type)
      'object-2
      'object-1))

(defun list-of-list->list (ll)
  (apply #'append ll))

(defun list->string (l)
  (mapcar #'write-to-string (list-of-list->list l)))

(defun get-motion-verb (transform len-objects)
  (if (eq len-objects 1)
      (gethash transform *reposition-hash*)
      (gethash transform *transport-hash*)))


(defun get-action-description (objects main-object-type transform)
  (setf *generated-description* "")
  (if (> (length objects) 1)
      (setf *generated-description* "put ")
      (setf *generated-description*
            (nth (random 2) (list "move " "shift "))))
  (setf *generated-description* (concatenate 'string *generated-description* "the "))
  (setf *generated-description* (concatenate 'string *generated-description* (write-to-string main-object-type) " "))
  (setf *generated-description* (concatenate 'string *generated-description* (get-motion-verb transform (length objects))))
  (if (> (length objects) 1)
  (setf *generated-description* (concatenate 'string *generated-description* (write-to-string (find-auxillary-object-type objects main-object-type)))))
  (values *generated-description*))

(defun get-color (color-name)
  (gethash color-name *color-hash*))
  
(defun write-motor-command (objects main-object-type transform auxillary-object-type)
    (let* ((motor-command (list->string objects))
         (mc-file (concatenate 'string "/home/crammel/JEPS_data/" "motor.txt")))
    (setf motor-command (append motor-command (list (write-to-string main-object-type)
                                                    (get-transform-string transform)
                                                    (write-to-string auxillary-object-type))))
      (format nil "~{~a~^ ~}" motor-command)))

(defun get-transform-string (transform)
  (gethash transform *transformation-hash*))

(defun write-motor-command-2 (objects main-object-type transform auxillary-object-type)
  (let* ((mc (write-to-string objects)))
    (setf mc (concatenate 'string mc " " (write-to-string main-object-type)))
    (setf mc (concatenate 'string mc (get-transform-string transform)))
    (setf mc (concatenate 'string mc (write-to-string auxillary-object-type)))
    (setf mc (concatenate 'string "(motor-program `" mc))
    (setf mc (concatenate 'string mc ")"))
    (values mc)))



(defun get-object-dimensions (object)
  (if (not (eq (type-of object) 'symbol))
      (setf object (get-obj-name object)))
  (cl-bullet:bounding-box-dimensions
   (btr:aabb
    (first
     (btr:rigid-bodies
      (btr:object btr:*current-bullet-world* object))))))

(defun get-object-height (object)
  (* 1.0 (cl-tf:z (get-object-dimensions object))))

(defun get-object-width (object)
  (* 0.7 (cl-tf:y (get-object-dimensions object))))

(defun get-object-depth (object)
  (* 0.7 (cl-tf:x (get-object-dimensions object))))

(defun get-translation-matrix (x y z)
 (inverse-transformation (cl-tf:make-transform-stamped "base_footprint" "base_footprint" 0.0 (cl-tf:make-3d-vector x y z) (cl-tf:make-quaternion 0.0 0.0 0.0 1.0)))
  )

(defun *leftward-transformation* (pose &optional (y 0.3))
  ;;(print pose)
  (cl-tf:pose->pose-stamped "base_footprint" 0.0
  (apply-transformation
   (get-translation-matrix 0 (- 0 y) 0 ) pose)))

(defun *rightward-transformation* (pose &optional (y 0.3))
  ;;(print pose)
  (cl-tf:pose->pose-stamped "base_footprint" 0.0
  (apply-transformation
   (get-translation-matrix 0 y 0 ) pose)))

(defun *forward-transformation* (pose &optional (x 0.15))
  ;;(print pose)
  (cl-tf:pose->pose-stamped "base_footprint" 0.0
  (apply-transformation
   (get-translation-matrix x 0 0 ) pose)))

(defun *backward-transformation* (pose &optional (x 0.15))
  ;;(print pose)
  (cl-tf:pose->pose-stamped "base_footprint" 0.0
  (apply-transformation
   (get-translation-matrix (- 0 x) 0 0 ) pose)))

(defun *on-transformation* (pose &optional (z 0.15))
  ;;(print pose)
  (cl-tf:pose->pose-stamped "base_footprint" 0.0
  (apply-transformation
   (get-translation-matrix 0 0 z ) pose)))

(defun top-of-object (object)
  (get-object-height object))


(defun find-object (?object-type)
  (let* ((possible-look-directions `(,*downward-look-coordinate*
                                     ,*left-downward-look-coordinate*
                                     ,*auxillary-look-coordinate*
                                     ,*right-downward-look-coordinate*))
         (?looking-direction (first possible-look-directions)))
    (setf possible-look-directions (rest possible-look-directions))
    ;; Look towards the first direction
    (perform (an action
                 (type looking)
                 (target (a location 
                            (pose ?looking-direction)))))
 
    ;; perception-object-not-found is the error that we get when the robot cannot find the object.
    ;; Now we're wrapping it in a failure handling clause to handle it
    (handle-failure perception-object-not-found
        ;; Try the action
        ((perform (an action
                      (type detecting)
                      (object (an object 
                                  (type ?object-type))))))
 
      ;; If the action fails, try the following:
      ;; try different look directions until there is none left.
      (when possible-look-directions
        (print "Perception error happened! Turning head.")
        ;; Resetting the head to look forward before turning again
        (perform (an action
                     (type looking) 
                     (direction forward)))
        (setf ?looking-direction (first possible-look-directions))
        (setf possible-look-directions (rest possible-look-directions))
        (perform (an action 
                     (type looking)
                     (target (a location
                                (pose ?looking-direction)))))
        ;; This statement retries the action again
        (cpl:retry))
      ;; If everything else fails, error out
      ;; Reset the neck before erroring out
      (perform (an action
                   (type looking)
                   (direction forward)))
      (cpl:fail 'object-nowhere-to-be-found))))



(defun pick-up-object (?object-type ?perceived-object ?grasping-arm) ;;Muhammed added a new parameter value: object-type
  (let* ((?possible-grasps (list-defined-grasps ?object-type))  ;define all possible grasps ;;Muhammed changed from hardcoded grasps
         (?remaining-grasps (copy-list ?possible-grasps))           ;make a copy to work though when trying each grasp
         (?grasp (first ?remaining-grasps)))                        ;this is the first one to try
 
      (cpl:with-retry-counters ((arm-change-retry 1))               ;there is one alternative arm if the first one fails
 
         ;; Outer handle-failure handling arm change
         (handle-failure object-unreachable
            ((setf ?remaining-grasps (copy-list ?possible-grasps))  ;make sure to try all possible grasps for each arm
             (setf ?grasp (first ?remaining-grasps))                ;get the first grasp to try
             (setf ?remaining-grasps (rest ?remaining-grasps))	    ;update the remaining grasps to try
 
            ;; Inner handle-failure handling grasp change
            (handle-failure (or manipulation-pose-unreachable gripper-closed-completely)
               ;; Try to perform the pick up
               ((perform (an action
                             (type picking-up)
                             (arm ?grasping-arm)
                             (grasp ?grasp)
                             (object ?perceived-object))))
 
               ;; When pick-up fails this block gets executed 
               ;;(format t "~%Grasping failed with ~a arm and ~a grasp~%" ?grasping-arm ?grasp)
               ;;(format t "~%Error: ~a~%" e)                       ;uncomment to see the error message
 
               ;; Check if we have any remaining grasps left.
               ;; If yes, then the block nested to it gets executed, which will
               ;; set the grasp that is used to the new value and trigger retry
 
               (when (first ?remaining-grasps)                      ;if there is a grasp remaining
                  (setf ?grasp (first ?remaining-grasps))           ;get it
                  (setf ?remaining-grasps (rest ?remaining-grasps)) ;update the remaining grasps to try
	          ;;(format t "Retrying with ~a arm and ~a grasp~%" ?grasping-arm ?grasp)
                  (park-arms)
                  (cpl:retry))
                ;; This will get executed when there are no more elements in the 
                ;; ?possible-grasps list. We print the error message and throw a new error
                ;; which will be caught by the outer handle-failure
                (print  "No more grasp retries left :(")
                (cpl:fail 'object-unreachable)))
 
             ;; This is the failure management of the outer handle-failure call
             ;; It changes the arm that is used to grasp
             ;;(format t "Manipulation failed with the ~a arm"?grasping-arm)
             ;; (print e)                                           ;uncomment if you want to see the error
             ;; Here we use the retry counter we defined. The value is decremented automatically
             (cpl:do-retry arm-change-retry
                ;; if the current grasping arm is right set left, else set right
                (setf ?grasping-arm (if (eq ?grasping-arm :right) 
                                        :left
                                        :right))
                (park-arms)
                (cpl:retry))
             ;; When all retries are exhausted print the error message.
             (print "No more arm change retries left :("))))
  ?grasping-arm) ; function value is the arm that was used to grasp the object
  


(defun motor-program (objects main-object-type transform auxillary-object-type &key (robot-location *base-pose-near-table*)
                                                   (distance 0.2))
  ;;(setup-generation)
  (setf *object-1-params* nil)
  (setf *object-2-params* nil)
  ;;(print (length objects))
  (setf *object-1-params* (first objects))
  (if (> (length objects) 1)
      (setf *object-2-params* (second objects)))
  ;;(setup-generation)

  (print *object-1-params*)
  (print *object-2-params*)
  (spawn-object (gethash (third *object-1-params*) *pose-hash*)
                (first *object-1-params*)
                'object-1
                (get-color (second *object-1-params*)))

  (if (> (length objects) 1)
  (spawn-object (gethash (third *object-2-params*) *pose-hash*)
                (first *object-2-params*)
                'object-2
                (get-color (second *object-2-params*))))
 (let ((desc-file (concatenate 'string "/home/crammel/JEPS_data/" "description.txt")))
  (with-open-file (str desc-file
                         :direction :output
                         :if-exists :append
                         :if-does-not-exist :create)
      
    (format str "~S@ ~S~%" *simulation-id* (get-action-description objects main-object-type transform))))
 (let ((mc-file (concatenate 'string "/home/crammel/JEPS_data/" "motor.txt")))
  (with-open-file (str mc-file
                         :direction :output
                         :if-exists :append
                         :if-does-not-exist :create)
      
    (format str "~S@ ~S~%" *simulation-id* (write-motor-command objects main-object-type transform auxillary-object-type))))

  ;;(setf (gethash #'backward-of *transport-hash*) "behind")
  (with-simulated-robot
    (let ((?navigation-goal robot-location))
      (cpl:seq
        ;; Moving the robot near the table.
        (perform (an action
                     (type going)
                     (target (a location 
                                (pose ?navigation-goal)))))
        (perform (a motion
                    (type moving-torso) 
                    (joint-angle 0.3)))
        (park-arms)
        ))
 
    (let ((?perceived-object (find-object main-object-type))
          (?grasping-arm :right))
      ;; We update the value of ?grasping-arm according to what the method used
      ;;(print ?perceived-object)

      (setf ?grasping-arm (pick-up-object main-object-type ?perceived-object ?grasping-arm))
      (print "--------------------------------------")
      (print (get-object-dimensions ?perceived-object))
      (park-arm ?grasping-arm)
      (sleep 0.5)
      (setf *final-destination* (obj-int:get-object-pose ?perceived-object))
      (if (> (length objects) 1)
          (setf *final-destination*
                (obj-int:get-object-pose
                 (find-object
                  (find-auxillary-object-type objects main-object-type)))))
      (print distance)
      (if (> (length objects) 1)
      (progn
        (if (eq transform #'*on-transformation*)
       (setf distance
             (get-object-height
              (find-auxillary-object-name objects main-object-type))))
        (if (or (eq transform #'*leftward-transformation*) (eq transform #'*rightward-transformation*))
       (setf distance
             (get-object-width
              (find-auxillary-object-name objects main-object-type))))        
        (if (or (eq transform #'*forward-transformation*) (eq transform #'*backward-transformation*))
       (setf distance
             (get-object-depth
              (find-auxillary-object-name objects main-object-type)))))
      (progn
       ; (if (eq transform #'*on-transformation*)
       ;(setf distance
       ;      (get-object-height 'object-1)))
        (if (or (eq transform #'*leftward-transformation*) (eq transform #'*rightward-transformation*))
       (setf distance
             (get-object-width 'object-1)))        
        (if (or (eq transform #'*forward-transformation*) (eq transform #'*backward-transformation*))
       (setf distance
             (get-object-depth 'object-1)))))        
      ;;(print distance)
      ;;(print *final-destination*)
      (let ((?drop-pose (apply transform (list *final-destination* distance))))
        (print ?drop-pose)
        (perform (an action
                     (type placing)
                     (arm ?grasping-arm)
                     (object ?perceived-object)
                     (target (a location 
                                (pose ?drop-pose))))))
      ;;(sleep 0.5)

      (park-arm ?grasping-arm)
      (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
          (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "10")))
      )))
