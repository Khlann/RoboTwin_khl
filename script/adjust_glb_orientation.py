#!/usr/bin/env python3
"""
GLB 文件方向调整工具
用于可视化并调整 GLB 文件的方向，支持交互式旋转和保存
"""

import argparse
import sys
import numpy as np
import trimesh
from pathlib import Path
import json

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("警告: 未安装 open3d，将使用 trimesh 的简单可视化")
    print("建议安装: pip install open3d")


class GLBOrientationAdjuster:
    """GLB 文件方向调整器"""
    
    def __init__(self, glb_path: str):
        self.glb_path = Path(glb_path)
        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB 文件不存在: {glb_path}")
        
        self.rotation_x = 0.0  # 绕 X 轴旋转（度）
        self.rotation_y = 0.0  # 绕 Y 轴旋转（度）
        self.rotation_z = 0.0  # 绕 Z 轴旋转（度）
        
        self.translation_x = 0.0  # X 轴平移（米）
        self.translation_y = 0.0  # Y 轴平移（米）
        self.translation_z = 0.0  # Z 轴平移（米）
        
        self.load_mesh()
    
    def load_mesh(self):
        """加载 GLB 文件"""
        print(f"正在加载 GLB 文件: {self.glb_path}")
        try:
            # 使用与代码库相同的方式加载
            with open(str(self.glb_path), "rb") as file_obj:
                loaded = trimesh.load(file_obj, file_type="glb")
            
            # 如果是场景，获取所有几何体
            if isinstance(loaded, trimesh.Scene):
                # 合并场景中的所有几何体
                meshes = []
                for name, geometry in loaded.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        meshes.append(geometry)
                    elif isinstance(geometry, (trimesh.PointCloud, trimesh.Path)):
                        # 跳过点云和路径
                        continue
                    else:
                        # 尝试转换为网格
                        try:
                            if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                                meshes.append(geometry)
                        except:
                            pass
                
                if len(meshes) == 0:
                    raise ValueError("场景中没有找到有效的网格")
                elif len(meshes) == 1:
                    self.mesh = meshes[0]
                else:
                    # 合并多个网格
                    print(f"找到 {len(meshes)} 个网格，正在合并...")
                    self.mesh = trimesh.util.concatenate(meshes)
            elif isinstance(loaded, trimesh.Trimesh):
                self.mesh = loaded
            else:
                # 尝试包装为场景
                scene = trimesh.Scene([loaded])
                self.mesh = loaded if isinstance(loaded, trimesh.Trimesh) else list(scene.geometry.values())[0]
            
            print(f"成功加载网格: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.faces)} 个面")
            print(f"边界框: {self.mesh.bounds}")
            print(f"中心点: {self.mesh.centroid}")
            
        except Exception as e:
            raise ValueError(f"加载 GLB 文件失败: {e}")
    
    def apply_transform(self, mesh=None):
        """应用旋转和平移变换"""
        if mesh is None:
            mesh = self.mesh.copy()
        
        # 转换为弧度
        rx = np.radians(self.rotation_x)
        ry = np.radians(self.rotation_y)
        rz = np.radians(self.rotation_z)
        
        # 创建旋转矩阵 (按 ZYX 顺序)
        # 先绕 Z 轴，再绕 Y 轴，最后绕 X 轴
        Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
        Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        
        # 组合旋转矩阵
        R = trimesh.transformations.concatenate_matrices(Rz, Ry, Rx)
        
        # 创建平移矩阵
        T_translation = trimesh.transformations.translation_matrix([
            self.translation_x,
            self.translation_y,
            self.translation_z
        ])
        
        # 应用旋转（相对于中心点）
        center = mesh.centroid
        T_center = trimesh.transformations.translation_matrix(-center)
        T_back = trimesh.transformations.translation_matrix(center)
        
        # 组合变换：先旋转（相对于中心），再平移
        transform = trimesh.transformations.concatenate_matrices(
            T_translation, T_back, R, T_center
        )
        
        mesh.apply_transform(transform)
        return mesh
    
    def apply_rotation(self, mesh=None):
        """应用旋转变换（保持向后兼容）"""
        return self.apply_transform(mesh)
    
    def visualize_with_open3d(self):
        """使用 Open3D 进行交互式可视化"""
        if not HAS_OPEN3D:
            print("Open3D 未安装，使用 trimesh 可视化")
            return self.visualize_with_trimesh()
        
        print("\n=== Open3D 交互式可视化 ===")
        print("操作说明:")
        print("  - 鼠标左键拖拽: 旋转视角")
        print("  - 鼠标右键拖拽: 平移")
        print("  - 滚轮: 缩放")
        print("\n旋转调整快捷键:")
        print("  - Q/A: 绕 X 轴旋转 +/-5°")
        print("  - W/S: 绕 Y 轴旋转 +/-5°")
        print("  - E/D: 绕 Z 轴旋转 +/-5°")
        print("  - 1/2: 绕 X 轴旋转 +/-1°（精细调整）")
        print("  - 3/4: 绕 Y 轴旋转 +/-1°（精细调整）")
        print("  - 5/6: 绕 Z 轴旋转 +/-1°（精细调整）")
        print("\n平移调整快捷键:")
        print("  - I/K: X 轴平移 +/-0.01m")
        print("  - J/L: Y 轴平移 +/-0.01m")
        print("  - U/O: Z 轴平移 +/-0.01m")
        print("  - Shift+I/K: X 轴平移 +/-0.001m（精细调整）")
        print("  - Shift+J/L: Y 轴平移 +/-0.001m（精细调整）")
        print("  - Shift+U/O: Z 轴平移 +/-0.001m（精细调整）")
        print("\n其他快捷键:")
        print("  - R: 重置视角")
        print("  - 0: 重置所有旋转角度为 0")
        print("  - T: 重置所有平移为 0")
        print("  - P: 打印当前旋转和平移")
        print("  - 空格: 保存当前变换并退出")
        print("  - ESC: 退出不保存")
        print("\n当前变换:")
        print(f"  旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}°")
        print(f"  平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
        
        # 创建 Open3D 可视化
        vis = o3d.visualization.VisualizerWithKeyCallback()
        window_name = f"GLB 方向调整: {self.glb_path.name}"
        vis.create_window(window_name=window_name, width=1024, height=768)
        
        # 应用当前变换（旋转+平移）
        rotated_mesh = self.apply_transform()
        
        # 转换为 Open3D 格式
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(rotated_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(rotated_mesh.faces)
        
        # 计算法线（Open3D 需要法线来正确渲染）
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.compute_triangle_normals()
        
        # 处理颜色：安全地检查不同类型的 visual 对象
        try:
            # 尝试获取顶点颜色
            if hasattr(rotated_mesh.visual, 'vertex_colors'):
                vertex_colors = rotated_mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    colors = vertex_colors[:, :3] / 255.0
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                else:
                    raise AttributeError("No vertex colors")
            else:
                raise AttributeError("No vertex_colors attribute")
        except (AttributeError, TypeError, IndexError):
            # 如果没有顶点颜色，尝试其他方法
            try:
                # 尝试从材质获取颜色
                if hasattr(rotated_mesh.visual, 'material'):
                    material = rotated_mesh.visual.material
                    if hasattr(material, 'main_color') and material.main_color is not None:
                        main_color = material.main_color
                        if isinstance(main_color, (list, np.ndarray)) and len(main_color) >= 3:
                            colors = np.tile(np.array(main_color[:3]) / 255.0, (len(rotated_mesh.vertices), 1))
                            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                        else:
                            raise AttributeError("Invalid main_color")
                    elif hasattr(material, 'diffuse') and material.diffuse is not None:
                        diffuse = material.diffuse
                        if isinstance(diffuse, (list, np.ndarray)) and len(diffuse) >= 3:
                            colors = np.tile(np.array(diffuse[:3]), (len(rotated_mesh.vertices), 1))
                            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                        else:
                            raise AttributeError("Invalid diffuse")
                    else:
                        raise AttributeError("No color in material")
                else:
                    raise AttributeError("No material")
            except (AttributeError, TypeError, IndexError):
                # 默认使用灰色
                default_color = [0.7, 0.7, 0.7]
                colors = np.tile(default_color, (len(rotated_mesh.vertices), 1))
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # 添加坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        vis.add_geometry(o3d_mesh)
        vis.add_geometry(coord_frame)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True  # 显示背面
        render_option.light_on = True  # 启用光照
        
        # 设置视角 - 根据模型大小自动调整
        bounds = rotated_mesh.bounds
        size = np.max(bounds[1] - bounds[0])  # 最大尺寸
        centroid = rotated_mesh.centroid
        
        ctr = vis.get_view_control()
        # 设置相机位置，距离模型适当距离
        distance = size * 2.5
        ctr.set_front([0, 0, 1])
        ctr.set_lookat(centroid)
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)
        
        # 打印调试信息
        print(f"\n模型信息:")
        print(f"  边界框: {bounds}")
        print(f"  中心点: {centroid}")
        print(f"  尺寸: {size:.3f}")
        print(f"  顶点数: {len(rotated_mesh.vertices)}")
        print(f"  面数: {len(rotated_mesh.faces)}")
        print(f"\n如果看不到模型，请尝试:")
        print(f"  - 鼠标滚轮缩放")
        print(f"  - 鼠标右键拖拽平移")
        print(f"  - 按 R 键重置视角")
        
        # 保存状态（默认不保存）
        save_and_exit = [False]
        rotation_step_large = 5.0
        rotation_step_small = 1.0
        translation_step_large = 0.01  # 0.01 米
        translation_step_small = 0.001  # 0.001 米（精细调整）
        
        # 注意：关闭窗口时也会正常退出，此时 save_and_exit[0] 仍为 False（不保存）
        
        def update_mesh():
            """更新网格显示"""
            nonlocal o3d_mesh, rotated_mesh
            rotated_mesh = self.apply_transform()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(rotated_mesh.vertices)
            # 重新计算法线
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.compute_triangle_normals()
            vis.update_geometry(o3d_mesh)
            # 注意：Open3D 不支持动态更新窗口标题，所以我们在终端打印
            # 窗口标题在创建时已设置，这里只更新终端输出
        
        def rotate_x_plus(vis):
            self.rotation_x += rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_x_minus(vis):
            self.rotation_x -= rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_y_plus(vis):
            self.rotation_y += rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_y_minus(vis):
            self.rotation_y -= rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_z_plus(vis):
            self.rotation_z += rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_z_minus(vis):
            self.rotation_z -= rotation_step_large
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_x_plus_small(vis):
            self.rotation_x += rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_x_minus_small(vis):
            self.rotation_x -= rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_y_plus_small(vis):
            self.rotation_y += rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_y_minus_small(vis):
            self.rotation_y -= rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_z_plus_small(vis):
            self.rotation_z += rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def rotate_z_minus_small(vis):
            self.rotation_z -= rotation_step_small
            update_mesh()
            print(f"旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}° | 平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        # 平移调整函数
        def translate_x_plus(vis):
            self.translation_x += translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_x_minus(vis):
            self.translation_x -= translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_y_plus(vis):
            self.translation_y += translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_y_minus(vis):
            self.translation_y -= translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_z_plus(vis):
            self.translation_z += translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_z_minus(vis):
            self.translation_z -= translation_step_large
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        # 精细平移调整函数
        def translate_x_plus_small(vis):
            self.translation_x += translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_x_minus_small(vis):
            self.translation_x -= translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_y_plus_small(vis):
            self.translation_y += translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_y_minus_small(vis):
            self.translation_y -= translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_z_plus_small(vis):
            self.translation_z += translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def translate_z_minus_small(vis):
            self.translation_z -= translation_step_small
            update_mesh()
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def reset_rotation(vis):
            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = 0.0
            update_mesh()
            print("已重置所有旋转角度为 0")
            print(f"旋转角度: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}°")
            return False
        
        def reset_translation(vis):
            self.translation_x = 0.0
            self.translation_y = 0.0
            self.translation_z = 0.0
            update_mesh()
            print("已重置所有平移为 0")
            print(f"平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
            return False
        
        def reset_view(vis):
            """重置视角到初始位置"""
            ctr = vis.get_view_control()
            bounds = rotated_mesh.bounds
            centroid = rotated_mesh.centroid
            ctr.set_front([0, 0, 1])
            ctr.set_lookat(centroid)
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.7)
            print("视角已重置")
            return False
        
        def print_transform(vis):
            print(f"\n当前变换:")
            print(f"  旋转: X={self.rotation_x:.2f}°, Y={self.rotation_y:.2f}°, Z={self.rotation_z:.2f}°")
            print(f"  平移: X={self.translation_x:.4f}m, Y={self.translation_y:.4f}m, Z={self.translation_z:.4f}m")
            return False
        
        def save_and_quit(vis):
            save_and_exit[0] = True
            print("\n保存当前旋转并退出...")
            return False
        
        def quit_without_save(vis):
            save_and_exit[0] = False
            print("\n退出，不保存更改...")
            return False
        
        # 注册键盘回调 - 大角度调整
        vis.register_key_callback(ord('Q'), rotate_x_plus)      # X轴 +5°
        vis.register_key_callback(ord('A'), rotate_x_minus)    # X轴 -5°
        vis.register_key_callback(ord('W'), rotate_y_plus)     # Y轴 +5°
        vis.register_key_callback(ord('S'), rotate_y_minus)    # Y轴 -5°
        vis.register_key_callback(ord('E'), rotate_z_plus)     # Z轴 +5°
        vis.register_key_callback(ord('D'), rotate_z_minus)    # Z轴 -5°
        
        # 注册键盘回调 - 小角度调整 (使用数字键进行精细调整)
        vis.register_key_callback(ord('1'), rotate_x_plus_small)   # X轴 +1°
        vis.register_key_callback(ord('2'), rotate_x_minus_small)  # X轴 -1°
        vis.register_key_callback(ord('3'), rotate_y_plus_small)   # Y轴 +1°
        vis.register_key_callback(ord('4'), rotate_y_minus_small)  # Y轴 -1°
        vis.register_key_callback(ord('5'), rotate_z_plus_small)   # Z轴 +1°
        vis.register_key_callback(ord('6'), rotate_z_minus_small)   # Z轴 -1°
        
        # 注册键盘回调 - 平移调整
        vis.register_key_callback(ord('I'), translate_x_plus)      # X轴 +0.01m
        vis.register_key_callback(ord('K'), translate_x_minus)     # X轴 -0.01m
        vis.register_key_callback(ord('J'), translate_y_plus)     # Y轴 +0.01m
        vis.register_key_callback(ord('L'), translate_y_minus)     # Y轴 -0.01m
        vis.register_key_callback(ord('U'), translate_z_plus)      # Z轴 +0.01m
        vis.register_key_callback(ord('O'), translate_z_minus)     # Z轴 -0.01m
        
        # 注册键盘回调 - 精细平移调整（使用小写字母）
        vis.register_key_callback(ord('i'), translate_x_plus_small)   # X轴 +0.001m
        vis.register_key_callback(ord('k'), translate_x_minus_small)  # X轴 -0.001m
        vis.register_key_callback(ord('j'), translate_y_plus_small)   # Y轴 +0.001m
        vis.register_key_callback(ord('l'), translate_y_minus_small)  # Y轴 -0.001m
        vis.register_key_callback(ord('u'), translate_z_plus_small)   # Z轴 +0.001m
        vis.register_key_callback(ord('o'), translate_z_minus_small)  # Z轴 -0.001m
        
        # 其他功能
        vis.register_key_callback(ord('R'), reset_view)        # 重置视角
        vis.register_key_callback(ord('0'), reset_rotation)    # 重置旋转
        vis.register_key_callback(ord('T'), reset_translation) # 重置平移
        vis.register_key_callback(ord('P'), print_transform)  # 打印变换信息
        vis.register_key_callback(ord(' '), save_and_quit)     # 空格：保存并退出
        vis.register_key_callback(27, quit_without_save)        # ESC：退出不保存
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        return save_and_exit[0]
    
    def visualize_with_trimesh(self):
        """使用 trimesh 进行简单可视化"""
        print("\n=== Trimesh 可视化 ===")
        print("当前变换:")
        print(f"  旋转: X={self.rotation_x:.1f}° Y={self.rotation_y:.1f}° Z={self.rotation_z:.1f}°")
        print(f"  平移: X={self.translation_x:.4f}m Y={self.translation_y:.4f}m Z={self.translation_z:.4f}m")
        
        # 应用变换
        transformed_mesh = self.apply_transform()
        
        # 显示
        transformed_mesh.show()
        
        return True
    
    def interactive_adjust(self):
        """交互式调整旋转和平移"""
        print("\n=== 交互式变换调整 ===")
        print("选择调整方式:")
        print("  1. 使用 Open3D 可视化窗口进行实时调整（推荐）")
        print("  2. 在终端中输入旋转和平移")
        print()
        
        if HAS_OPEN3D:
            choice = input("请选择 (1/2，默认1): ").strip()
            if choice == '' or choice == '1':
                # 使用 Open3D 可视化调整
                print("\n正在打开 Open3D 可视化窗口...")
                saved = self.visualize_with_open3d()
                return saved
            else:
                # 终端输入方式
                return self._terminal_adjust()
        else:
            return self._terminal_adjust()
    
    def _terminal_adjust(self):
        """终端输入方式调整"""
        print("\n=== 终端输入调整 ===")
        print("输入旋转角度（度）和平移（米），格式: rx ry rz tx ty tz")
        print("例如: 90 0 0 0.01 0 0  (绕 X 轴旋转 90 度，X 轴平移 0.01 米)")
        print("也可以只输入旋转: rx ry rz")
        print("输入 'show' 查看当前效果")
        print("输入 'save' 保存并退出")
        print("输入 'quit' 退出不保存")
        print()
        
        while True:
            try:
                cmd = input("请输入命令 (x y z / show / save / quit): ").strip()
                
                if cmd.lower() == 'quit' or cmd.lower() == 'q':
                    print("退出，未保存更改")
                    return False
                
                if cmd.lower() == 'save' or cmd.lower() == 's':
                    return True
                
                if cmd.lower() == 'show':
                    if HAS_OPEN3D:
                        saved = self.visualize_with_open3d()
                        if saved:
                            return True
                    else:
                        self.visualize_with_trimesh()
                    continue
                
                # 解析旋转角度和平移
                parts = cmd.split()
                if len(parts) == 3:
                    # 只输入旋转角度
                    self.rotation_x = float(parts[0])
                    self.rotation_y = float(parts[1])
                    self.rotation_z = float(parts[2])
                    print(f"设置旋转角度: X={self.rotation_x}°, Y={self.rotation_y}°, Z={self.rotation_z}°")
                elif len(parts) == 6:
                    # 输入旋转和平移
                    self.rotation_x = float(parts[0])
                    self.rotation_y = float(parts[1])
                    self.rotation_z = float(parts[2])
                    self.translation_x = float(parts[3])
                    self.translation_y = float(parts[4])
                    self.translation_z = float(parts[5])
                    print(f"设置变换:")
                    print(f"  旋转: X={self.rotation_x}°, Y={self.rotation_y}°, Z={self.rotation_z}°")
                    print(f"  平移: X={self.translation_x:.4f}m, Y={self.translation_y:.4f}m, Z={self.translation_z:.4f}m")
                else:
                    print("格式错误，请输入三个数字（旋转）或六个数字（旋转+平移）")
                    print("例如: 90 0 0  或  90 0 0 0.01 0 0")
                    continue
                
                # 询问是否查看
                view = input("是否查看效果? (y/n): ").strip().lower()
                if view == 'y':
                    if HAS_OPEN3D:
                        saved = self.visualize_with_open3d()
                        if saved:
                            return True
                    else:
                        self.visualize_with_trimesh()
                    
            except KeyboardInterrupt:
                print("\n退出，未保存更改")
                return False
            except Exception as e:
                print(f"错误: {e}")
    
    def save_rotated_glb(self, output_path: str = None):
        """保存旋转后的 GLB 文件"""
        if output_path is None:
            # 默认保存为原文件名_rotated.glb
            output_path = self.glb_path.parent / f"{self.glb_path.stem}_rotated{self.glb_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # 应用变换（旋转+平移）
        transformed_mesh = self.apply_transform()
        
        # 创建场景
        scene = trimesh.Scene([transformed_mesh])
        
        # 导出为 GLB
        print(f"\n{'='*60}")
        print(f"正在保存变换后的 GLB 文件...")
        print(f"原始文件: {self.glb_path}")
        print(f"保存位置: {output_path.absolute()}")
        print(f"{'='*60}")
        scene.export(str(output_path))
        print(f"✓ GLB 文件已成功保存!")
        
        # 保存变换信息到 JSON
        transform_info = {
            "original_file": str(self.glb_path.absolute()),
            "output_file": str(output_path.absolute()),
            "rotation_degrees": {
                "x": self.rotation_x,
                "y": self.rotation_y,
                "z": self.rotation_z
            },
            "rotation_radians": {
                "x": np.radians(self.rotation_x),
                "y": np.radians(self.rotation_y),
                "z": np.radians(self.rotation_z)
            },
            "translation_meters": {
                "x": self.translation_x,
                "y": self.translation_y,
                "z": self.translation_z
            }
        }
        
        info_path = output_path.parent / f"{output_path.stem}_transform_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(transform_info, f, indent=2, ensure_ascii=False)
        print(f"✓ 变换信息已保存到: {info_path.absolute()}")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="GLB 文件方向调整工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 交互式调整
  python adjust_glb_orientation.py assets/objects/hammer_igen.glb
  
  # 直接应用旋转并保存
  python adjust_glb_orientation.py assets/objects/hammer_igen.glb --rotate 90 0 0 --output hammer_rotated.glb
  
  # 仅可视化，不保存
  python adjust_glb_orientation.py assets/objects/hammer_igen.glb --visualize-only
        """
    )
    
    parser.add_argument(
        'glb_file',
        type=str,
        help='GLB 文件路径'
    )
    
    parser.add_argument(
        '--rotate', '-r',
        nargs=3,
        type=float,
        metavar=('X', 'Y', 'Z'),
        help='直接应用旋转角度（度），格式: X Y Z'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件路径（默认: 原文件名_rotated.glb）'
    )
    
    parser.add_argument(
        '--visualize-only', '-v',
        action='store_true',
        help='仅可视化，不保存'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建调整器
        adjuster = GLBOrientationAdjuster(args.glb_file)
        
        if args.visualize_only:
            # 仅可视化
            if HAS_OPEN3D:
                adjuster.visualize_with_open3d()
            else:
                adjuster.visualize_with_trimesh()
        elif args.rotate:
            # 直接应用旋转
            adjuster.rotation_x = args.rotate[0]
            adjuster.rotation_y = args.rotate[1]
            adjuster.rotation_z = args.rotate[2]
            print(f"应用旋转: X={adjuster.rotation_x}°, Y={adjuster.rotation_y}°, Z={adjuster.rotation_z}°")
            adjuster.save_rotated_glb(args.output)
        else:
            # 交互式调整
            if adjuster.interactive_adjust():
                adjuster.save_rotated_glb(args.output)
    
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
