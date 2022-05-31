from distutils import command
from numpy.lib.function_base import disp
import open3d as o3d
import open3d
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
from sklearn import neighbors

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray)")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def display_inlier_outlier_twoPCD(inlier_cloud, outlier_cloud):

    print("Showing outliers (red) and inliers (gray)")
    pcd1 = o3d.geometry.PointCloud(inlier_cloud)
    pcd2 = o3d.geometry.PointCloud(outlier_cloud)
    pcd2.paint_uniform_color([1, 0, 0])
    pcd1.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    o3d.visualization.draw_geometries([pcd1, pcd2])


######################## UI ########################
# Start Button
def radiusButton_Click():
    filePath = filePath_label_text['text']
    if filePath == "":
        statusUpdate("Please select file !")
        return
    statusUpdate("")
    filePath = os.path.relpath(filePath)
    param1 = param2 = 0
    if Radius1.get(1.0, tk.END+"-1c") != '':
        param1 = int(Radius1.get(1.0, tk.END+"-1c"))
    if Radius2.get(1.0, tk.END+"-1c") != '':    
        param2 = float(Radius2.get(1.0, tk.END+"-1c"))
    pcd = root.currentPCD
    cl, ind = pcd.remove_radius_outlier(param1, param2)
    display_inlier_outlier(pcd, ind)
    pcd = pcd.select_by_index(ind)
    root.currentPCD = pcd   

def radius2Button_Click():
    filePath = filePath_label_text['text']
    if filePath == "":
        statusUpdate("Please select file !")
        return
    statusUpdate("")
    filePath = os.path.relpath(filePath)
    root.filePath = filePath
    param1 = param2 = 0
    if Radius1.get(1.0, tk.END+"-1c") != '':
        param1 = int(Radius1.get(1.0, tk.END+"-1c"))
    if Radius2.get(1.0, tk.END+"-1c") != '':    
        param2 = float(Radius2.get(1.0, tk.END+"-1c"))
    removedPCD = o3d.geometry.PointCloud()
    removedPCD_ind = list()
    originPCD = root.currentPCD
    count = 0
    param = float(Radius3.get(1.0, tk.END+"-1c"))
    while True:
        pcd = root.currentPCD
        cl, ind = pcd.remove_radius_outlier(param1, param2)
        count += 1
        currentRemoved = pcd.select_by_index(ind, invert=True)
        removedPCD += currentRemoved
        removedPCD_ind += ind
        pcd = pcd.select_by_index(ind)
        print("Removed PCD : {:>7}, Origin PCD : {:>8}, R/O : {:.6f}".format(len(np.asarray(currentRemoved.points)) , len(np.asarray(originPCD.points)) , len(np.asarray(currentRemoved.points)) / len(np.asarray(originPCD.points))))
        if len(np.asarray(currentRemoved.points)) / len(np.asarray(originPCD.points)) < param:
            root.currentPCD = pcd
            break
        root.currentPCD = pcd
    print("Radius Outlier Removal : {} times".format(count))
    display_inlier_outlier_twoPCD(root.currentPCD, removedPCD)

def statis2Button_Click():
    filePath = filePath_label_text['text']
    if filePath == "":
        statusUpdate("Please select file !")
        return
    statusUpdate("")
    filePath = os.path.relpath(filePath)
    param1 = param2 = 0
    if Radius1.get(1.0, tk.END+"-1c") != '':
        param1 = int(Radius1.get(1.0, tk.END+"-1c"))
    if Radius2.get(1.0, tk.END+"-1c") != '':    
        param2 = float(Radius2.get(1.0, tk.END+"-1c"))
    removedPCD = o3d.geometry.PointCloud()
    removedPCD_ind = list()
    originPCD = root.currentPCD
    count = 0
    param = float(Statistical3.get(1.0, tk.END+"-1c"))
    while True:
        pcd = root.currentPCD
        cl, ind = pcd.remove_statistical_outlier(param1, param2)
        count += 1
        currentRemoved = pcd.select_by_index(ind, invert=True)
        removedPCD += currentRemoved
        removedPCD_ind += ind
        pcd = pcd.select_by_index(ind)
        print("Removed PCD : {:>7}, Origin PCD : {:>8}, R/O : {:.6f}".format(len(np.asarray(removedPCD.points)) , len(np.asarray(originPCD.points)) , len(np.asarray(removedPCD.points)) / len(np.asarray(originPCD.points))))
        if len(np.asarray(removedPCD.points)) / len(np.asarray(originPCD.points)) > param:
            root.currentPCD = pcd
            break
        root.currentPCD = pcd
    print("Statis Outlier Removal : {} times".format(count))
    display_inlier_outlier_twoPCD(root.currentPCD, removedPCD)

def statisButton_Click():
    filePath = filePath_label_text['text']
    if filePath == "":
        statusUpdate("Please select file !")
        return
    statusUpdate("")
    filePath = os.path.relpath(filePath)
    param1 = param2 = 0
    if Radius1.get(1.0, tk.END+"-1c") != '':
        param1 = int(Statistical1.get(1.0, tk.END+"-1c"))
    if Radius2.get(1.0, tk.END+"-1c") != '':    
        param2 = float(Statistical2.get(1.0, tk.END+"-1c"))
    pcd = root.currentPCD
    cl, ind = pcd.remove_statistical_outlier(param1, param2)
    display_inlier_outlier(pcd, ind)
    pcd = pcd.select_by_index(ind)
    root.currentPCD = pcd   

def viewBtn_Click():
    o3d.visualization.draw_geometries([root.currentPCD])

def saveBtn_Click():
    output_file = root.filePath[:-4]+"_denoised.ply"
    csv = root.filePath[:-4]+"_denoised.csv"
    o3d.io.write_point_cloud(output_file, root.currentPCD)
    data = np.asarray(root.currentPCD.points)
    color = np.asarray(root.currentPCD.colors)
    pcdData = np.concatenate((data, color), axis=1)
    np.savetxt(csv, pcdData, delimiter=",", header="X,Y,Z,R,G,B", comments="")
    statusUpdate("Data Saved !")

def menuLoad_Click():
    filePath = filedialog.askopenfilename(parent=root, initialdir='./', filetypes = (("csv files","*.csv"),("txt files","*.txt"),("all files","*.*")))
    filePath_label_text['text'] = filePath
    filePath = os.path.relpath(filePath)
    root.filePath = filePath
    fileType = filePath[len(filePath)-3:]
    pcd = o3d.geometry.PointCloud()
    if fileType == "csv":
        np.set_printoptions(suppress=True)
        Data = np.loadtxt(filePath, dtype=np.float64, skiprows=1, delimiter=',', usecols=(0,1,2), unpack=False)
        Color = np.loadtxt(filePath, dtype=np.float64, skiprows=1, delimiter=',', usecols=(3,4,5), unpack=False)
        pcd.points =  o3d.utility.Vector3dVector(Data)
        pcd.colors = o3d.utility.Vector3dVector(Color)
    elif fileType == "txt":
        pcd = o3d.io.read_point_cloud(filePath, format='xyzrgb')
    else:
        statusUpdate("Unsupported file type !!")
    root.currentPCD = pcd
    statusUpdate("Data Loaded !")

def statusUpdate(s):
    Status_label_text['text'] = s

ctypes.windll.shcore.SetProcessDpiAwareness(0)
root = tk.Tk()
root.title('removeOutlier')
root.geometry('520x130')
filePath = ""
root.currentPCD = o3d.geometry.PointCloud()
root.filePath = ""

# Create Menu
root.theSegments = None
root.maxIndex = 0
menuBar = tk.Menu(root)
fileMenu = tk.Menu(menuBar)
fileMenu = Menu(menuBar, tearoff=0)
fileMenu.add_command(label="Load", command=menuLoad_Click)
menuBar.add_cascade(label="File", menu=fileMenu)
root.config(menu=menuBar)
# Create Label
filePath_label = tk.Label(root, text=' 檔案位置 :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
filePath_label.place(x=5, y=7)
Radius_label = tk.Label(root, text="    Radius    :", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
Radius_label.place(x=5, y=67)
Statistical_label = tk.Label(root, text=" Statistical :", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
Statistical_label.place(x=5, y=37)
filePath_label_text = tk.Label(root, text='', font=('Microsoft JhengHei UI', 10, 'bold'))
filePath_label_text.place(x=80, y=7)
Status_label = tk.Label(root, text=' Status :  ', font=('Microsoft JhengHei UI', 10, 'bold'))
Status_label.place(x=5, y=97)
Status_label_text = tk.Label(root, text='', font=('Microsoft JhengHei UI', 10, 'bold'))
Status_label_text.place(x=70, y=97)
neighbors_label = tk.Label(root, text='neighbors', font=('Microsoft JhengHei UI', 10, 'bold'))
neighbors_label.place(x=120, y=67)
ratio_label = tk.Label(root, text='radius', font=('Microsoft JhengHei UI', 10, 'bold'))
ratio_label.place(x=250, y=67)
neighbors2_label = tk.Label(root, text='neighbors', font=('Microsoft JhengHei UI', 10, 'bold'))
neighbors2_label.place(x=120, y=37)
ratio_label = tk.Label(root, text='ratio', font=('Microsoft JhengHei UI', 10, 'bold'))
ratio_label.place(x=250, y=37)

# Create Button
RadiusBtn = tk.Button(root, command=radiusButton_Click,      text=" Once ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
RadiusBtn.place(x=400, y=67)
Radius2Btn = tk.Button(root, command=radius2Button_Click,      text=" Multi ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
Radius2Btn.place(x=450, y=67)
StatisticalBtn = tk.Button(root, command=statisButton_Click, text=" Once ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
StatisticalBtn.place(x=400, y=37)
Statistical2Btn = tk.Button(root, command=statis2Button_Click, text=" Multi ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
Statistical2Btn.place(x=450, y=37)
viewBtn = tk.Button(root, command=viewBtn_Click,             text=" View ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
viewBtn.place(x=450, y=97)
saveBtn = tk.Button(root, command=saveBtn_Click,             text=" Save ", font=('Microsoft JhengHei UI', 9, 'bold'),background='white smoke')
saveBtn.place(x=400, y=97)
# Create input box
Radius1 = tk.Text(root, height=1, width=5)
Radius2 = tk.Text(root, height=1, width=5)
Radius3 = tk.Text(root, height=1, width=7)
Radius1.insert(END, '30')
Radius2.insert(END, '0.10')
Radius3.insert(END, '0.0001')
Radius1.place(x=80, y=70)
Radius2.place(x=200, y=70)
Radius3.place(x=300, y=70)
Statistical1 = tk.Text(root, height=1, width=5)
Statistical2 = tk.Text(root, height=1, width=5)
Statistical3 = tk.Text(root, height=1, width=7)
Statistical1.place(x=80, y=40)
Statistical2.place(x=200, y=40)
Statistical3.place(x=300, y=40)
Statistical1.insert(END,'20')
Statistical2.insert(END, '2.0')
Statistical3.insert(END, '0.3')
####################################################


if __name__ == '__main__':
    root.mainloop()
