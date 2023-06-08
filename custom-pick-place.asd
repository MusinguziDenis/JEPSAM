(defsystem custom-pick-place

  :depends-on (roslisp-utilities ; for ros-init-function

               cl-transforms
               cl-transforms-stamped
               cl-tf
               cram-tf

               cram-language
               cram-executive
               cram-designators
               cram-prolog
               cram-projection
               cram-occasions-events
               cram-utilities ; for force-ll etc

               cram-common-failures

               cram-physics-utils ; for reading "package://" paths
               cl-bullet ; for handling BOUNDING-BOX datastructures
               cram-bullet-reasoning
               cram-bullet-reasoning-belief-state
               cram-bullet-reasoning-utilities

               cram-location-costmap
               cram-btr-visibility-costmap
               ;; cram-semantic-map-costmap
               cram-robot-pose-gaussian-costmap
               cram-occupancy-grid-costmap
               ;;cram-btr-spatial-relations-costmap

               cram-pr2-projection ; for projection process modules
               ;;cram-mobile-pick-place-plans
               cram-pr2-description
               cram-knowrob-pick-place)

  :components
  ((:module "src"
    :components
    ((:file "package")

     ;; actions such as REACHING, LIFTING, GRASPING, GRIPPING, LOOKING-AT, etc.
     (:file "atomic-action-plans" :depends-on ("package"))
     (:file "atomic-action-designators" :depends-on ("package" "atomic-action-plans"))
     (:file "tutorial" :depends-on ("package"))
     (:file "setup" :depends-on ("package"))
     ;; PICKING-UP and PLACING actions
     (:file "pick-place-plans" :depends-on ("package" "atomic-action-designators"))
     (:file "pick-place-designators" :depends-on ("package"
                                                  "pick-place-plans"))

     ;; high-level plans such as DRIVE-AND-PICK-UP, PERCEIVE, etc.
     (:file "generate-data" :depends-on ("package" "tutorial" "setup" "atomic-action-designators" "pick-place-designators"))
     (:file "generate-samples" :depends-on ("package" "tutorial" "setup" "atomic-action-designators" "pick-place-designators"))
))))
