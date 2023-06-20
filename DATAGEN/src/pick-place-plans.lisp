(in-package :pp-cust)

(defparameter *first-camera-position* nil)
(defparameter *second-camera-position* nil)
(cpl:def-cram-function pick-up (?object-designator
                                ?arm ?gripper-opening ?grip-effort ?grasp
                                ?left-reach-poses ?right-reach-poses
                                ?left-grasping-poses ?right-grasping-poses
                                ?left-lift-poses ?right-lift-poses)
  (cram-tf:visualize-marker (obj-int:get-object-pose ?object-designator)
                            :r-g-b-list '(1 1 0) :id 300)

  (setf *first-camera-position* (obj-int:get-object-pose ?object-designator))

  (let ((?looking-direction *first-camera-position*))
    (perform (an action
                 (type looking)
                 (target (a location 
                            (pose ?looking-direction))))))

  (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (ensure-directories-exist simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "0")))
  (cpl:par
    ;;(roslisp:ros-info (pick-place pick-up) "Opening gripper")
    (exe:perform
     (desig:an action
               (type setting-gripper)
               (gripper ?arm)
               (position ?gripper-opening)))
    ;;(roslisp:ros-info (pick-place pick-up) "Reaching")
    (cpl:with-failure-handling
        ((common-fail:manipulation-low-level-failure (e)
           (roslisp:ros-warn (pp-plans pick-up)
                             "Manipulation messed up: ~a~%Ignoring."
                             e)
            (return)
           ))
        (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
          (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "1")))
      ;;(print "printing")
      ;;(print (type-of ?left-reach-poses))
      ;;(print (type-of ?right-reach-poses))
      ;;(print (type-of (first ?right-reach-poses)))
      ;;(print ?left-reach-poses)
      ;;(print ?right-reach-poses)
      (exe:perform
       (desig:an action
                 (type reaching)
                 (left-poses ?left-reach-poses)
                 (right-poses ?right-reach-poses)))))
  (cpl:with-failure-handling
      ((common-fail:manipulation-low-level-failure (e)
         (roslisp:ros-warn (pp-plans pick-up)
                           "Manipulation messed up: ~a~%Ignoring."
                           e)
         (return)
         ))
      (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "2")))

    (exe:perform
     (desig:an action
               (type grasping)
               (left-poses ?left-grasping-poses)
               (right-poses ?right-grasping-poses))))
  ;;(roslisp:ros-info (pick-place pick-up) "Gripping")
    (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "3")))
  ;;(print ?grip-effort)
  (exe:perform
   (desig:an action
             (type gripping)
             (gripper ?arm)
             (effort ?grip-effort)
             (object ?object-designator)))
  ;;(roslisp:ros-info (pick-place pick-up) "Assert grasp into knowledge base")
  (cram-occasions-events:on-event
   (make-instance 'cpoe:object-attached
     :object-name (desig:desig-prop-value ?object-designator :name)
     :arm ?arm))
  ;;(roslisp:ros-info (pick-place pick-up) "Lifting")
  (cpl:with-failure-handling
      ((common-fail:manipulation-low-level-failure (e)
         (roslisp:ros-warn (pp-plans pick-up)
                           "Manipulation messed up: ~a~%Ignoring."
                           e)
         (return)))
      (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "4")))
    ;;(print ?left-lift-poses)
    ;;(print ?right-lift-poses)
    (exe:perform
     (desig:an action
               (type lifting)
               (left-poses ?left-lift-poses)
               (right-poses ?right-lift-poses)))
      (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "5")))

    ))


(cpl:def-cram-function place (?object-designator
                              ?arm
                              ?left-reach-poses ?right-reach-poses
                              ?left-put-poses ?right-put-poses
                              ?left-retract-poses ?right-retract-poses
                              ?placing-location)

  (print "Placing location")
  (print ?placing-location)

  
  ;;(setf *second-camera-position* (first ?right-put-poses))
  (setf *second-camera-position* ?placing-location)

  (if (not *second-camera-position*)
      (progn
        (let ((?looking-direction *second-camera-position*))
            (perform (an action
                 (type looking)
                 (target (a location 
                            (pose ?looking-direction))))))))

  (roslisp:ros-info (pick-place place) "Reaching")
  
  (cpl:with-failure-handling
      ((common-fail:manipulation-low-level-failure (e)
         (roslisp:ros-warn (pp-plans pick-up)
                           "Manipulation messed up: ~a~%Ignoring."
                           e)
          (return)
         ))
    (exe:perform
     (desig:an action
               (type reaching)
               (left-poses ?left-reach-poses)
               (right-poses ?right-reach-poses))))
   (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (ensure-directories-exist simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "6")))
  (roslisp:ros-info (pick-place place) "Putting")
  (cpl:with-failure-handling
      ((common-fail:manipulation-low-level-failure (e)
      (roslisp:ros-warn (pp-plans pick-up)
                           "Manipulation messed up: ~a~%Ignoring."
                           e)
         (return)))
    (exe:perform
     (desig:an action
               (type putting)
               (left-poses ?left-put-poses)
               (right-poses ?right-put-poses))))
   (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (ensure-directories-exist simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "7")))
  (roslisp:ros-info (pick-place place) "Opening gripper")
  (exe:perform
   (desig:an action
             (type releasing)
             (gripper ?arm)))
   (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (ensure-directories-exist simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "8")))
  (roslisp:ros-info (pick-place place) "Retract grasp in knowledge base")
  (cram-occasions-events:on-event
   (make-instance 'cpoe:object-detached
     :arm ?arm
     :object-name (desig:desig-prop-value ?object-designator :name)))
  (roslisp:ros-info (pick-place place) "Retracting")
  (cpl:with-failure-handling
      ((common-fail:manipulation-low-level-failure (e)
         (roslisp:ros-warn (pp-plans pick-up)
                           "Manipulation messed up: ~a~%Ignoring."
                           e)
         (return)
         ))
    (exe:perform
     (desig:an action
               (type retracting)
               (left-poses ?left-retract-poses)
               (right-poses ?right-retract-poses))))
    
    ;; (if (not *second-camera-position*)
    ;;   (progn
    ;;     (let ((?looking-direction *second-camera-position*))
    ;;         (perform (an action
    ;;              (type looking)
    ;;              (target (a location 
    ;;                         (pose ?looking-direction))))))))

    (park-arm ?arm)

   (let ((simulation-dir (concatenate 'string "/home/crammel/JEPS_data/" *simulation-id* "/")))
    ;;(print simulation-dir)
    (ensure-directories-exist simulation-dir)
    (btr:png-from-camera-view :png-path (concatenate 'string simulation-dir "9")))
  )


;; (defun perform-phases-in-sequence (action-designator)
;;   (declare (type desig:action-designator action-designator))
;;   (let ((phases (desig:desig-prop-value action-designator :phases)))
;;     (mapc (lambda (phase)
;;             (format t "Executing phase: ~%~a~%~%" phase)
;;             (exe:perform phase))
;;           phases)))

;; (cpl:def-cram-function pick-up (action-designator object arm grasp)
;;   (perform-phases-in-sequence action-designator)
;;   (cram-occasions-events:on-event
;;    (make-instance 'cpoe:object-gripped :object object :arm arm :grasp grasp)))


;; (cpl:def-cram-function place (action-designator object arm)
;;   (perform-phases-in-sequence action-designator)
;;   (cram-occasions-events:on-event
;;    (make-instance 'cpoe:object-released :arm arm :object object)))

