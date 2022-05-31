from contextlib import suppress
from struct import unpack
from tokenize import group
from numpy.lib.function_base import disp
import open3d as o3d
import numpy as np
import pandas as pd
from tkinter import * 
from tkinter import messagebox
import matplotlib.pyplot as plt
import copy
import os
import math
import time
import tkinter as tk
from tkinter import Menu, filedialog
from tkinter.constants import DISABLED
from PIL import Image, ImageTk
import sys
import ctypes
import matplotlib.backends.backend_tkagg
import csv


sys.setrecursionlimit(5000)
if getattr(sys, 'frozen', False):
    ctypes.windll.kernel32.SetDllDirectoryW('D:/Anaconda/envs/pytorch/Library/bin')
    ctypes.CDLL('libiomp5md.dll')
    ctypes.CDLL('mkl_avx2.1.dll')
    ctypes.CDLL('mkl_core.1.dll')
    ctypes.CDLL('mkl_intel_thread.1.dll')
    ctypes.CDLL('mkl_def.1.dll')

    ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def removeOutlier(pcd, nb_neighbors=20, std_ratio=2.0, display = False):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    if display:
        display_inlier_outlier(pcd, ind)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def createPoissonMesh_segment(pcd, segments, filePath=""):
    print("Testing IO for meshes ...")
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Pcd and Mesh Visualization", 1024, 768)
    vis.show_settings = True
    time_start = time.time() #開始計時
    output_file = filePath[:-4]+"_mesh.ply"
    segments_count = len(segments)
    currentMesh = o3d.geometry.TriangleMesh()
    for i in range(segments_count):
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(segments[i], depth=8, scale=1.1, width=0, linear_fit=False)
        bbox = segments[i].get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)   
        geometry_name = "Mesh" + str(i)
        pcd_name = "Pcd" + str(i)
        maxPt = segments[i].get_min_bound()
        vis.add_3d_label([maxPt[0], maxPt[1], maxPt[2]], f"Plane {i}")
        currentMesh += p_mesh_crop
        vis.add_geometry(pcd_name, segments[i], group="PCD")
        vis.add_geometry(geometry_name, p_mesh_crop, group="Mesh")
    time_end = time.time()    #結束計時
    time_c= time_end - time_start   #執行所花時間
    print('time cost', time_c, 's')

    o3d.io.write_triangle_mesh(output_file, currentMesh)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run() 



def createPoissonMesh(input_file, output_file="", remove_outlier = False, down_sampling = False, removeLowDensity=True, visualize=False, visualizeDensity=False, saveSimplify = False):
    print("Testing IO for meshes ...")
    time_start = time.time() #開始計時
    pcd = o3d.io.read_point_cloud(input_file[:-4]+"_pcd.ply") # Read the point cloud
    if output_file =="":
        output_file = input_file[:-4]+"_mesh.ply"

    if remove_outlier:
        pcd = removeOutlier(pcd)
    
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)#[0]
    
    if removeLowDensity:
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    if visualizeDensity:
        visualizeDense(poisson_mesh, densities)
    #cropping
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    #Down sampling triangle mesh
    if down_sampling:
        
        dec_mesh = p_mesh_crop.simplify_quadric_decimation(100000)        

        #移除不必要的點        
        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()
        
        #export
        o3d.io.write_triangle_mesh(output_file, dec_mesh)
    else:
        o3d.io.write_triangle_mesh(output_file, p_mesh_crop)

    time_end = time.time()    #結束計時
    time_c= time_end - time_start   #執行所花時間
    print('time cost', time_c, 's') 
    #execution of function 是否做可視化
    if saveSimplify:
        output_path = output_file[:-4]
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        my_lods = lod_mesh_export(poisson_mesh, [100000,50000,10000,1000,100], ".ply", output_path)
    if visualize:
        visualizePcdMesh(output_file)

def xyzRgbToPly(input_file, output_file="", visualizeNormal=False, csv=True):
    if output_file =="":
        output_file = input_file[:-4]+"_pcd.ply"
    pcd = o3d.geometry.PointCloud()
    if csv:
        x = root.options.index(root.listOfVar[0].get())
        y = root.options.index(root.listOfVar[1].get())
        z = root.options.index(root.listOfVar[2].get())
        r = root.options.index(root.listOfVar[3].get())
        g = root.options.index(root.listOfVar[4].get())
        b = root.options.index(root.listOfVar[5].get())
        np.set_printoptions(suppress=True)
        Data = np.loadtxt(input_file, dtype=np.float64, skiprows=1, delimiter=',', usecols=(x,y,z), unpack=False)
        Color = np.loadtxt(input_file, dtype=np.float64, skiprows=1, delimiter=',', usecols=(r,g,b), unpack=False)
        if root.chkValue.get():
            Color[Color > 0] = 0
        pcd.points =  o3d.utility.Vector3dVector(Data)
        pcd.colors = o3d.utility.Vector3dVector(Color)
    else:
        pcd = o3d.io.read_point_cloud(input_file, format='xyzrgb')
    # add normals information in pcd

    pcd.estimate_normals(
        # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30) # Old One
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=250) # New One
    )
    
    # visualize the normal
    if visualizeNormal:
        o3d.visualization.draw_geometries([pcd],
                                        # zoom=0.3412,
                                        # front=[0.4257, -0.2125, -0.8795],
                                        # lookat=[2.6172, 2.0475, 1.532],
                                        # up=[-0.0694, -0.9768, 0.2024],
                                        point_show_normal=True)    

    o3d.io.write_point_cloud(output_file, pcd)
    return pcd

def plyNormal(input_file, output_file="", visualizeNormal=False):
    if output_file =="":
        output_file = input_file[:-4]+"_pcd.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    # add normals information in pcd

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # visualize the normal
    if visualizeNormal:
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024],
                                        point_show_normal=True)    

    o3d.io.write_point_cloud(output_file, pcd)
    return pcd

def xyzGrayToPly(input_file, output_file=""):
    if output_file =="":
        output_file = input_file[:-4]+"_pcd.ply"
    data = pd.read_csv(input_file, sep=" ", header=None)
    print(data)
    data.columns = ["X","Y","Z","G"]
    X = data["X"].to_numpy()
    Y = data["Y"].to_numpy()
    Z = data["Z"].to_numpy()
    xyz = np.asarray([X,Y,Z])
    xyz_t = np.transpose(xyz)
    print(xyz)
    print(xyz_t)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_t)
    o3d.io.write_point_cloud(output_file, pcd)

def visualizePcdMesh(input_file, filterMesh=False):
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    mesh = o3d.io.read_triangle_mesh(input_file) # Read the mesh
    if filterMesh:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
        mesh.compute_vertex_normals()

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Pcd and Mesh Visualization", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Pcd", pcd) 
    vis.add_geometry("Mesh", mesh) 
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def visualizeMesh(input_file):
    mesh = o3d.io.read_triangle_mesh(input_file)
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Mesh Visualization", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Mesh", mesh) 
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def visualizePcd(input_file):
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Point Cloud Visualization", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Pcd", pcd) 
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def visualizeDense(mesh, dense):
    densities = np.asarray(dense)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh],
                                    zoom=0.664,
                                    front=[-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101])

def calcPlanesDist(segment_models, planesThickness, name, coef, countingPlaneNum=-1):

    if countingPlaneNum!=-1:
        n = countingPlaneNum
    else:
        n = len(segment_models)

    print("="*50)
    print("以 Segment 中心切面算距離:")
    for i in range(n):
        for j in range(n):
            if i!=j and i<j:
                a1,b1,c1,d1 = segment_models[i]
                a2,b2,c2,d2 = segment_models[j]
                a = (a1+a2)/2
                b = (b1+b2)/2
                c = (c1+c2)/2
                distance = abs(d1-d2)/math.sqrt(a*a+b*b+c*c)
                print(f"Distance between plane {i} and plane {j} is {distance*1000:.3f} (miu m)")
    print("="*50)

    print("以 Segment 中心切面算距離並/材質係數:")
    totalThickness = 0.0
    for i in range(n-1):
        a1,b1,c1,d1 = segment_models[i]
        a2,b2,c2,d2 = segment_models[i+1]
        a = (a1+a2)/2
        b = (b1+b2)/2
        c = (c1+c2)/2
        distance = abs(d1-d2)/math.sqrt(a*a+b*b+c*c)
        distance /= coef[i]
        totalThickness+=distance
        print(f"Distance between plane {i} and plane {i+1} ({name[i]}) is {distance*1000:.3f} (miu m)")
    print("="*50)

    # 這一部分因為沒針對材質交界處也各取一半分別/材質係數所以有時算出來是負的(不過公司暫時還不需要看這結果)
    print("以 Segment 表面算距離並/材質係數:")
    for i in range(n-1):
        a1,b1,c1,d1 = segment_models[i]
        a2,b2,c2,d2 = segment_models[i+1]
        a = (a1+a2)/2
        b = (b1+b2)/2
        c = (c1+c2)/2
        distance = abs(d1-d2)/math.sqrt(a*a+b*b+c*c)-planesThickness[i]/2-planesThickness[i+1]/2
        distance /= coef[i]
        print(f"Distance between plane {i} and plane {i+1} ({name[i]})  is {distance*1000:.3f} (miu m)")
    print("="*50)

    print("以物件表面算間距:")
    for i in range(n-1):
        a1,b1,c1,d1 = segment_models[i]
        a2,b2,c2,d2 = segment_models[i+1]
        a = (a1+a2)/2
        b = (b1+b2)/2
        c = (c1+c2)/2
        distance = abs(d1-d2)/math.sqrt(a*a+b*b+c*c)-planesThickness[i]/2-planesThickness[i+1]/2
        print(f"Distance between plane {i} and plane {i+1} is {distance*1000:.3f} (miu m)")
    print("="*50)
    print(f"總厚度= {totalThickness*1000:.3f} (miu m)")
    print("="*50)
    
def applyNoise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def calcPlanesDist_noCoef(segment_models, planesThickness, countingPlaneNum=-1):

    if countingPlaneNum!=-1:
        n = countingPlaneNum
    else:
        n = len(segment_models)

    print("n = ", n)

    print("="*50)
    print("以 Segment 中心切面算距離:")
    for i in range(n):
        for j in range(n):
            if i!=j and i<j:
                a1,b1,c1,d1 = segment_models[i]
                a2,b2,c2,d2 = segment_models[j]
                a = (a1+a2)/2
                b = (b1+b2)/2
                c = (c1+c2)/2
                distance = (d1-d2)/math.sqrt(a*a+b*b+c*c)
                print(f"Distance between plane {i} and plane {j} is {distance*1000:.3f} (miu m)")
    print("="*50)

    totalThickness = 0.0
    for i in range(n-1):
        a1,b1,c1,d1 = segment_models[i]
        a2,b2,c2,d2 = segment_models[i+1]
        a = (a1+a2)/2
        b = (b1+b2)/2
        c = (c1+c2)/2
        distance = abs(d1-d2)/math.sqrt(a*a+b*b+c*c)
        totalThickness+=distance

    print("以物件表面算間距:")
    for i in range(n-1):
        a1,b1,c1,d1 = segment_models[i]
        a2,b2,c2,d2 = segment_models[i+1]
        a = (a1+a2)/2
        b = (b1+b2)/2
        c = (c1+c2)/2
        distance = (d1-d2)/math.sqrt(a*a+b*b+c*c)-planesThickness[i]/2-planesThickness[i+1]/2
        print(f"Distance between plane {i} and plane {i+1} is {distance*1000:.3f} (miu m)")
    print("="*50)
    print(f"總厚度= {totalThickness*1000:.3f} (miu m)")
    print("="*50)
    
def applyNoise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def calcDistAndShow(pcd, segment_models, segments, planesThickness, name, coef):
    calcPlanesDist(segment_models, planesThickness, name, coef, len(coef)+1)
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D -3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Origin Pcd", pcd)
    for i in range(len(segment_models)):
        print(f"Plane {i} thickness = {planesThickness[i]*1000:.3f} (miu m)")
        maxPt = segments[i].get_max_bound()
        vis.add_geometry(f"planes {i} Pcd", segments[i])
        vis.add_3d_label([maxPt[0], maxPt[1]+2, maxPt[2]+5], f"Plane {i}")
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()

def removeSegROI(rest, segment, averageHeight, findThickness=True, visualize=False):
    # o3d.visualization.draw_geometries([rest])
    mu, sigma = 0, 0.1  # mean and standard deviation
    # segment = applyNoise(segment, mu, sigma) # 為了求 bounding box，需要構成 8 個點，所以隨機產生一點雜訊不然可能會因為是平面吃 error
    segBbox = segment.get_oriented_bounding_box()
    segBbox.color = (1, 0, 0)
    dilateExtent = np.array([segBbox.extent[0]*1.3, segBbox.extent[1]*1.3, averageHeight])
    toRemoveBbox = o3d.geometry.OrientedBoundingBox(segBbox.center, segBbox.R, dilateExtent)
    if visualize:
        o3d.visualization.draw_geometries([rest, toRemoveBbox])
    toRemoveIndexes = toRemoveBbox.get_point_indices_within_bounding_box(rest.points)

    thickness = 0
    if findThickness:
        # 用比較小的範圍找出跟厚度相關的 Bounding box
        erosionExtent = np.array([segBbox.extent[0]/2, segBbox.extent[1]/2, averageHeight])
        ThicknessBbox = o3d.geometry.OrientedBoundingBox(segBbox.center, segBbox.R, erosionExtent)
        toRemoveSmallRoiIndexes = ThicknessBbox.get_point_indices_within_bounding_box(rest.points)
        planeRegionPcd = rest.select_by_index(toRemoveSmallRoiIndexes, invert=False)
        remove_outlier= True
        if remove_outlier:
            planeRegionPcd = removeOutlier(planeRegionPcd, nb_neighbors=40, std_ratio=0.5)

        planeRegionBbox = planeRegionPcd.get_oriented_bounding_box()
        if visualize:
            o3d.visualization.draw_geometries([rest, planeRegionBbox])
        thickness = planeRegionBbox.extent[2]
    
    rest = rest.select_by_index(toRemoveIndexes, invert=True)

    if visualize:
        o3d.visualization.draw_geometries([rest, toRemoveBbox])

    return rest,thickness

def sortingPlanes(segment_models, segments, planesThickness,reverse=True, n=-1):
    if n == -1:
        n = len(segments)    
    if not reverse:
        for i in range(n):
            for j in range(n-i-1):
                if segment_models[j][3] > segment_models[j+1][3]:
                    segments[j], segments[j+1] = segments[j+1], segments[j]
                    segment_models[j], segment_models[j+1] = segment_models[j+1], segment_models[j]
                    planesThickness[j], planesThickness[j+1] = planesThickness[j+1], planesThickness[j]
    else:
        for i in range(n):
            for j in range(n-i-1):
                if segment_models[j][3] < segment_models[j+1][3]:
                    segments[j], segments[j+1] = segments[j+1], segments[j]
                    segment_models[j], segment_models[j+1] = segment_models[j+1], segment_models[j]
                    planesThickness[j], planesThickness[j+1] = planesThickness[j+1], planesThickness[j]
    return segment_models, segments, planesThickness

def sortingPlanes_selection(segment_models, segments, planesThickness,reverse=True, n=-1):
    if n == -1:
        n = len(segments)    
    if not reverse:
        for i in range(n):
            minIdx = i
            for j in range(i+1, n):
                if segments[minIdx].get_min_bound()[2] < segments[j].get_min_bound()[2]:
                    minIdx = j
            segments[i], segments[minIdx] = segments[minIdx], segments[i]
            segment_models[i], segment_models[minIdx] = segment_models[minIdx], segment_models[i]
            planesThickness[i], planesThickness[minIdx] = planesThickness[minIdx], planesThickness[i]                    
    else: 
        for i in range(n):
            minIdx = i
            for j in range(i+1, n):
                if segments[minIdx].get_min_bound()[2] > segments[j].get_min_bound()[2]:
                    minIdx = j
            segments[i], segments[minIdx] = segments[minIdx], segments[i]
            segment_models[i], segment_models[minIdx] = segment_models[minIdx], segment_models[i]
            planesThickness[i], planesThickness[minIdx] = planesThickness[minIdx], planesThickness[i]  
    return segment_models, segments, planesThickness

def segmentPlanesByRansac(input_file, max_plane_idx=6, parallelRatio = 0.1, downSample=False, removeUselessPlane=False, visualizeFoundPlane=False, avoidVertical=False):
    pcd = o3d.io.read_point_cloud(input_file)
    if downSample:
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
    root.currentPCD = pcd
    maxBound = pcd.get_max_bound()
    minBound = pcd.get_min_bound()  

    segment_models={}
    segments={}
    averageHeight = (maxBound[1]-minBound[1] )/max_plane_idx/4
    idx = 0
    rest=pcd
    normalsToCmp = []
    planesThickness = []
    foundParallelNum = 0

    while foundParallelNum<max_plane_idx:
        if  len(np.asarray(rest.points)) < 100:
            break
        segment_models[idx], inliers = rest.segment_plane(distance_threshold=0.03,ransac_n=3,num_iterations=1000)
        [a, b, c, d] = segment_models[idx]
        print(f"Plane {idx} equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # 先找是否為 Vertical plane
        if avoidVertical and (abs(a-0)<0.1 and abs(b-1)<0.1 and abs(c-0)<0.1):
            print("Is vertical plane")
            if removeUselessPlane:
                not_parallel_plane = rest.select_by_index(inliers)
                rest, thickness = removeSegROI(rest, not_parallel_plane, averageHeight, findThickness=False, visualize=visualizeFoundPlane)
            continue
            
        # 先建立一個要比較的平面(之後取的面都要跟這個平行)
        elif len(normalsToCmp) == 0:
            normalsToCmp = segment_models[idx]
            
        # 平面若跟第一次找到的不是平行就先跳過
        else:
            dif = normalsToCmp[:2]-segment_models[idx][:2]
            if np.max(dif) - np.min(dif) > parallelRatio:
                print("not parallel")
                print(dif)
                if removeUselessPlane:
                    not_parallel_plane = rest.select_by_index(inliers)
                    rest, thickness = removeSegROI(rest, not_parallel_plane, averageHeight, findThickness=False, visualize=visualizeFoundPlane)                
                continue
        
        segments[idx]=rest.select_by_index(inliers)

        rest, thickness = removeSegROI(rest, segments[idx], averageHeight, findThickness=False, visualize=visualizeFoundPlane)
        planesThickness.append(thickness)
        idx+=1
        foundParallelNum+=1
        print("pass",idx,"/",max_plane_idx,"done.\n")        
    
    segment_models, segments, planesThickness = sortingPlanes(segment_models, segments, planesThickness, n=foundParallelNum)

    return pcd, segment_models, segments, planesThickness, foundParallelNum

######################## UI ########################
# Start Button
def startButton_Click():
    materialName = []
    filePath = filePath_label_text['text']
    if filePath == "":
        statusUpdate("Please select file !")
        return
    statusUpdate("")
    filePath = os.path.relpath(filePath)
    input_file = filePath
    if input_file[len(input_file)-3:] == "txt":
        pcd = xyzRgbToPly(input_file, csv=False)
    elif input_file[len(input_file)-3:] == "csv":
        pcd = xyzRgbToPly(input_file, csv=True)
    else:
        statusUpdate("Unsupported file type !!")
        return
    max_plane_idx = 7
    if layer.get(1.0, tk.END+"-1c") != '':
        max_plane_idx = int(layer.get(1.0, tk.END+"-1c"))
    pcd, segment_models, segments, planesThickness, foundParallelNum = segmentPlanesByRansac(input_file[:-4]+"_pcd.ply", max_plane_idx, removeUselessPlane=True, visualizeFoundPlane=False)
    # segment_models, segments, planesThickness = sortingPlanes(segment_models, segments, planesThickness)
    calcPlanesDist_noCoef(segment_models, planesThickness, countingPlaneNum=foundParallelNum)
    segment_models, segments, planesThickness = sortingPlanes_selection(segment_models, segments, planesThickness, n = foundParallelNum)
    createPoissonMesh_segment(pcd, segments, filePath)
    

def menuLoad_Click():
    filePath = filedialog.askopenfilename(parent=root, initialdir='./', filetypes = (("csv files","*.csv"),("txt files","*.txt"),("all files","*.*")))
    filePath_label_text['text'] = filePath
    filePath = os.path.relpath(filePath)
    list_of_column_names = []
    if filePath[len(filePath)-3:] == "csv":
        with open(filePath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for row in csv_reader:
                list_of_column_names = list(filter(None, row))
                options = list()
                for i in list_of_column_names:
                    j = i.replace(" ", "")
                    options.append(j)
                break
        setDropList(options)
        root.options = options
    statusUpdate("")

def statusUpdate(s):
    Status_label_text['text'] = s

def setDropList(options=""):
    root.optionLen = len(options)
    for item in root.Dropdown_list:
        item.destroy()
    if options != "":
        root.xList = options
        root.yList = options
        root.zList = options
        root.rList = options
        root.gList = options
        root.bList = options
    else:
        root.xList = ["( X )"]
        root.yList = ["( Y )"]
        root.zList = ["( Z )"]
        root.rList = ["( R )"]
        root.gList = ["( G )"]
        root.bList = ["( B )"]
    createDroplist()





ctypes.windll.shcore.SetProcessDpiAwareness(0)
root = tk.Tk()
root.title('createPoissonMesh')
root.geometry('700x130')
root.currentPCD = o3d.geometry.PointCloud()
filePath = ""
# Create Menu
menuBar = tk.Menu(root)
fileMenu = tk.Menu(menuBar)
fileMenu = Menu(menuBar, tearoff=0)
fileMenu.add_command(label="Load", command=menuLoad_Click)
menuBar.add_cascade(label="File", menu=fileMenu)
root.config(menu=menuBar)
# Create Label
filePath_label = tk.Label(root, text=' 檔案位置 :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
filePath_label.place(x=5, y=7)
filePath_label_text = tk.Label(root, text='', font=('Microsoft JhengHei UI', 10, 'bold'))
filePath_label_text.place(x=80, y=7)
Param_label = tk.Label(root, text=' Layers :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
Param_label.place(x=5, y=37)
Column_label = tk.Label(root, text=' Column :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
Column_label.place(x=5, y=67)
Status_label = tk.Label(root, text=' Status :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
Status_label.place(x=5, y=97)
Status_label_text = tk.Label(root, text='', font=('Microsoft JhengHei UI', 10, 'bold'))
Status_label_text.place(x=70, y=97)
Layer_label = tk.Label(root, text=' ', font=('Microsoft JhengHei UI', 10, 'bold'))
Layer_label.place(x=100, y=37)
# Create inputBox
layer = tk.Text(root, height=1, width=5)
layer.insert(END, '10')
layer.place(x=70, y=39)
# Create Button
loadButton = tk.Button(root, command=startButton_Click, text="Show", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
loadButton.place(relx=0.95, rely=0.8, anchor=tk.CENTER)
# Create CheckBox
root.chkValue = tk.BooleanVar() 
root.chkValue.set(False)
noColor = tk.Checkbutton(root, text='No Color', var=root.chkValue, background='#c6e2ff') 
noColor.place(x=550, y=66)
# Drop down list
root.xList = ["( X )"]
root.yList = ["( Y )"]
root.zList = ["( Z )"]
root.rList = ["( R )"]
root.gList = ["( G )"]
root.bList = ["( B )"]
root.optionLen = 1
root.options = list()

root.Dropdown_list = list()

def createDroplist():
    root.listOfVar = list()
    ShowBtn_list = list()
    max = root.optionLen

    listVar = tk.StringVar(root)
    listVar.set(root.xList[0])
    max -= 1
    opt = tk.OptionMenu(root, listVar, *root.xList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+0*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)
    listVar = tk.StringVar(root)
    if max:
        listVar.set(root.yList[1])
        max -= 1
    else:
        listVar.set(root.yList[root.optionLen-1])
    opt = tk.OptionMenu(root, listVar, *root.yList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+1*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)

    listVar = tk.StringVar(root)
    if max:
        listVar.set(root.zList[2])
        max -= 1
    else:
        listVar.set(root.zList[root.optionLen-1])
    opt = tk.OptionMenu(root, listVar, *root.zList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+2*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)

    listVar = tk.StringVar(root)
    if max:
        listVar.set(root.rList[3])
        max -= 1
    else:
        listVar.set(root.rList[root.optionLen-1])
    opt = tk.OptionMenu(root, listVar, *root.rList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+3*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)

    listVar = tk.StringVar(root)
    if max:
        listVar.set(root.gList[4])
        max -= 1
    else:
        listVar.set(root.gList[root.optionLen-1])
    opt = tk.OptionMenu(root, listVar, *root.gList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+4*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)

    listVar = tk.StringVar(root)
    if max:
        listVar.set(root.bList[5])
        max -= 1
    else:
        listVar.set(root.bList[root.optionLen-1])
    opt = tk.OptionMenu(root, listVar, *root.bList)
    opt.config(width=4, font=('Microsoft JhengHei UI', 9))
    opt.place(x=85+5*75, y=62)
    root.Dropdown_list.append(opt)
    root.listOfVar.append(listVar)

createDroplist()

####################################################

if __name__ == '__main__':

    # Start
    root.mainloop()
