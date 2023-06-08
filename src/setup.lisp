
(in-package :pp-cust)

;; roslaunch cram_pick_place_tutorial world.launch

(defun init-projection ()
  (def-fact-group costmap-metadata ()
    (<- (location-costmap:costmap-size 12 12))
    (<- (location-costmap:costmap-origin -6 -6))
    (<- (location-costmap:costmap-resolution 0.05))

    (<- (location-costmap:costmap-padding 0.2))
    (<- (location-costmap:costmap-manipulation-padding 0.2))
    (<- (location-costmap:costmap-in-reach-distance 0.6))
    (<- (location-costmap:costmap-reach-minimal-distance 0.2)))

  (setf cram-bullet-reasoning-belief-state:*robot-parameter* "robot_description")
  (setf cram-bullet-reasoning-belief-state:*kitchen-parameter* "kitchen_description")

  ;; (sem-map:get-semantic-map)

  (cram-occasions-events:clear-belief)

  (setf cram-tf:*tf-default-timeout* 2.0)

  (setf prolog:*break-on-lisp-errors* t))

(roslisp-utilities:register-ros-init-function init-projection)
