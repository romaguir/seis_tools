import pandas
import numpy as np
from gmtpy import GMT
from seis_tools.models.geometry import cart2polar
import pickle
#from scipy.io import read_array
#from scipy import zeros, linspace, shape, Float, concatenate

#--------------------------------------------------------------------------------
def plot_vtk_slice(vtk_slice,theta_range=[0,360],depth_range=[0,2885],**kwargs): 
#--------------------------------------------------------------------------------

   '''
   This will plot a cross section of a .vtk file.  Open the .vtk file in paraview,
   choose a slice, and select 'File -> export data'.  This will save the data as
   a .csv file, which can be plotted with this function.

   args--------------------------------------------------------------------------
   vtk_slice: the .csv file
   theta_range: the range in degrees you wish to plot. 
                dtype=tuple
                default=[0,360] (whole earth)
   depth_range: the range in depth you wish to plot.
                dtype=tuple
                default=[0,2885] (whole mantle)

   kwargs------------------------------------------------------------------------
   cmap = colormap to use
          dtype=string
          default='BlueWhiiteOrangeRed'

   data_range = max and min values to use in colorbar.
                dtype=tuple
                default=[-1.0,1.0]

   csteps = number of divisions in colorbar
            dtype=int
            default=100

   cmap_direction = forward or reverse colormap
                    dtype=string
                    default='i'

   fname = filename
           dtype=string
           default='slice.pdf'

   rotation = number of degrees to rotate figure
            dtype=float
            default=90.0 (i.e., rotate from lat coor to colat based coor)

   contour = True or False
   '''

   #get kwargs-------------------------------------------------------------------
   cmap_dir = '/geo/home/romaguir/utils/colors/'
   cmap = kwargs.get('cmap','BlueWhiteOrangeRed')
   cmap = cmap_dir+cmap
   data_range = kwargs.get('data_range',[-0.25,0.25])
   csteps = kwargs.get('csteps',100)
   cmap_direction=kwargs.get('cmap_direction','i')
   fname = kwargs.get('fname','slice.pdf')
   rotation=kwargs.get('rotation',90.0)
   contour=kwargs.get('contour',True)

   #read csv slice (output of paraview)------------------------------------------
   f = pandas.read_csv(vtk_slice)
   p1 = f['Points:0']
   p2 = f['Points:1']
   p3 = f['Points:2']
   dvp = f['dVp()']

   #transform to polar, earth coordinates----------------------------------------
   r,t = cart2polar(p1,p3)
   r *= 6371.0
   t=np.degrees(t)
   

   print min(dvp),max(dvp)

   #setup GMT plot---------------------------------------------------------------
   gmt = GMT(config={'BASEMAP_TYPE':'fancy',
                      'HEADER_FONT_SIZE':'14'})
   region = '{}/{}/{}/{}'.format(theta_range[0],theta_range[1],6371.0-depth_range[1],6371.0-depth_range[0])
   surf_region = '{}/{}/{}/{}'.format(theta_range[0]-2,theta_range[1]+2,6371.0-depth_range[1],6500.0-depth_range[0])
   scale = 'P6i' #Polar, 8 inch radius
   cptfile = gmt.tempfilename()
   grdfile = gmt.tempfilename()

   #gmt.makecpt(C=cmap,T='{}/{}/{}'.format(data_range[0],data_range[1],csteps),Z=False,out_filename=cptfile,D=cmap_direction)
   gmt.makecpt(C=cmap,T='-0.25/0.25/0.025',A=100,out_filename=cptfile,D=True)

   gmt.surface(in_columns=[t+rotation,r,dvp],G=grdfile,I='0.5/25',T=0.0,R=surf_region,out_discard=True)

   '''
   #plot the data----------------------------------------------------------------
   gmt.psxy( R=region,
             J=scale,
             #B='a15f15:"":/200f200:""::."":WSne',
             B='a300f300',
             S='s0.20',
             #G='blue',
             C=cptfile,
             in_columns=[t+rotation,r,dvp] )
   '''
   #plot the data----------------------------------------------------------------
   gmt.grdimage( grdfile, 
                 R=region,
                 J=scale,
                 B='a300f300',
                 C=cptfile,
                 E='i5' )
   
   #contour the data-------------------------------------------------------------
   if contour == True:
      gmt.grdcontour( grdfile,
                      R=region,
                      J=scale,
                      C=cptfile,
                      W='1' )

   #plot 660---------------------------------------------------------------------
   d660 = np.loadtxt('/geo/home/romaguir/utils/660_polar.dat')
   print d660
   gmt.psxy( R=region,
             J=scale,
             W='1',
             in_rows=[d660] )
            
   gmt.save(fname)

def extract_tomo_profile(filename,delta,**kwargs):
   '''  
   Takes in an xyz slice of a tomographic model (example output of crossect_180), and 
   returns a depth profile at a specified arc distance, delta.
   '''  
   middle = kwargs.get('middle',False)

   try:
      slice = np.loadtxt(filename)
   except ValueError:
      slice = np.loadtxt(filename,skiprows=1)

   x = slice[:,0]
   y = slice[:,1]
   z = slice[:,2]
   rad = []
   val = []

   if middle:
      x_0 = x[0]
      x_e = x[::-1][0]
      x_m = (x_0 + x_e)/2.0
      print 'x_0,x_e,x_m',x_0,x_e,x_m
      for i in range(0,len(x)-1):
         if x[i] <= x_m and x[i+1] > x_m:
            rad.append(y[i])
            val.append(z[i])
   else:
      for i in range(0,len(x)):
         if x[i] == delta:
            rad.append(y[i])
            val.append(z[i])

   return np.array((rad,val))

def cpt2seg(file_name, sym=False, discrete=False):
    """Reads a .cpt palette and returns a segmented colormap.

    sym : If True, the returned colormap contains the palette and a mirrored copy.
    For example, a blue-red-green palette would return a blue-red-green-green-red-blue colormap.

    discrete : If true, the returned colormap has a fixed number of uniform colors.
    That is, colors are not interpolated to form a continuous range.

    Example :
    >>> _palette_data = cpt2seg('palette.cpt')
    >>> palette = matplotlib.colors.LinearSegmentedColormap('palette', _palette_data, 100)
    >>> imshow(X, cmap=palette)

    Funciton URL http://matplotlib.1069221.n5.nabble.com/function-to-create-a-colormap-from-cpt-palette-td2165.html
    """
   
   
    dic = {}
    f = open(file_name, 'r')
    rgb = read_array(f)
    rgb = rgb/255.
    s = shape(rgb)
    colors = ['red', 'green', 'blue']
    for c in colors:
        i = colors.index(c)
        x = rgb[:, i+1]

        if discrete:
            if sym:
                dic[c] = zeros((2*s[0]+1, 3), dtype=Float)
                dic[c][:,0] = linspace(0,1,2*s[0]+1)
                vec = concatenate((x ,x[::-1]))
            else:
                dic[c] = zeros((s[0]+1, 3), dtype=Float)
                dic[c][:,0] = linspace(0,1,s[0]+1)
                vec = x
            dic[c][1:, 1] = vec
            dic[c][:-1,2] = vec
               
        else:
            if sym:
                dic[c] = zeros((2*s[0], 3), dtype=Float)
                dic[c][:,0] = linspace(0,1,2*s[0])
                vec = concatenate((x ,x[::-1]))
            else:
                dic[c] = zeros((s[0], 3), dtype=Float)
                dic[c][:,0] = linspace(0,1,s[0])
                vec = x
            dic[c][:, 1] = vec
            dic[c][:, 2] = vec
   
    return dic

def pk_to_cpt(pk,file_name):
   '''
   pickled matplotlib colorplot to cpt
   '''

   colors = pickle.load(file(pk)) 
   outfile = open(file_name,'w')

   #write header
   outfile.write('# '+file_name+'\n')
   outfile.write('# Ross Maguire '+'\n')
   outfile.write('# COLOR MODEL = RGB'+'\n')

   #layer = np.arange(-1.0*len(colors)/2,len(colors)/2)
   #print layer
   #print len(layer)

   for i in range(0,len(colors)-1):
      outfile.write('{} {:3.0f} {:3.0f} {:3.0f} {} {:3.0f} {:3.0f} {:3.0f}'.format(i,colors[i][0]*255,colors[i][1]*255,colors[i][2]*255,
                                                                            i+1,colors[i+1][0]*255,colors[i+1][1]*255,colors[i+1][2]*255))
      outfile.write('\n')

   outfile.write('B 0 0 0'+'\n')       #background black
   outfile.write('F 255 255 255'+'\n') #foreground white
   outfile.write('N 255 0 0'+'\n')
