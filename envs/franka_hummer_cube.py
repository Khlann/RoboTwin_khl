from ._base_task import Base_Task
from .utils import *
import sapien
from ._GLOBAL_CONFIGS import *


class franka_hummer_cube(Base_Task):

    def setup_demo(self, **kwags):
        # Store task-specific config BEFORE calling _init_task_env_ (which calls load_actors)
        self.cube_pose_config = kwags.get("cube_pose", None)
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Create hammer at a fixed position suitable for single arm workspace
        # Position adjusted for single arm: closer to center, within reach
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0.0, -0.05, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        
        # Get cube pose from config or use random generation
        if self.cube_pose_config is not None:
            # Use fixed pose from config
            # Format: {"position": [x, y, z], "quaternion": [w, x, y, z]} or {"position": [x, y, z], "quaternion": [x, y, z, w]}
            pos = self.cube_pose_config.get("position", [0, 0, 0.76])
            quat = self.cube_pose_config.get("quaternion", [1, 0, 0, 0])
            # Ensure quaternion is in [w, x, y, z] format
            if len(quat) == 4:
                # Check if it's [x, y, z, w] format (first element is small)
                if abs(quat[0]) < 0.1 and abs(quat[3]) > 0.9:
                    # Likely [x, y, z, w], convert to [w, x, y, z]
                    quat = [quat[3], quat[0], quat[1], quat[2]]
            cube_pose = sapien.Pose(pos, quat)
        else:
            # Use random pose generation (default behavior)
            # Create cube with reduced randomization scope for single arm workspace
            # Single arm has limited reach, so we constrain the cube to a smaller area
            # xlim: [-0.15, 0.15] (reduced from [-0.25, 0.25])
            # ylim: [0.0, 0.12] (reduced from [-0.05, 0.15], moved forward for better reach)
            cube_pose = rand_pose(
                xlim=[-0.15, 0.15],  # Reduced range for single arm
                ylim=[0.0, 0.12],    # Reduced and moved forward
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )
            
            # Ensure cube is not too close to center or too far
            while abs(cube_pose.p[0]) < 0.03 or np.sum(pow(cube_pose.p[:2], 2)) < 0.001:
                cube_pose = rand_pose(
                    xlim=[-0.15, 0.15],
                    ylim=[0.0, 0.12],
                    zlim=[0.76],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.5],
                )

        # Create cube for collision and planning (used for functional points and contact detection)
        self.cube = create_box(
            scene=self,
            pose=cube_pose,
            half_size=(0.025, 0.025, 0.012),
            color=(1, 0, 0),
            name="cube",
            is_static=True,
        )
        
        # Create 121_igen_box for visual display only (no collision, overlaps with cube)
        self.visual_box = create_actor(
            scene=self,
            pose=cube_pose,  # Same pose as cube to overlap
            modelname="121_igen_box",
            convex=False,
            is_static=True,
            model_id=0,
            no_collision=True,  # No collision, visual only
        )
        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            cube_pose.p[0] - 0.05,
            cube_pose.p[1] - 0.05,
            cube_pose.p[0] + 0.05,
            cube_pose.p[1] + 0.05,
        ])

    def play_once(self):
        # Single arm task: always use left arm (no arm selection logic)
        arm_tag = ArmTag("left")

        # Grasp the hammer with the left arm
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        
        # Move the hammer upwards
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        # Place the hammer on the cube's functional point (position 1) to beat it
        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.cube.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            ))

        self.info["info"] = {"{A}": "020_hammer/base0", "{B}": "cube", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        cube_pose = self.cube.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])
        return np.all(abs(hammer_target_pose[:2] - cube_pose[:2]) < eps) and self.check_actors_contact(
            self.hammer.get_name(), self.cube.get_name())
