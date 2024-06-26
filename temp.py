


import geopandas as gpd
import matplotlib.pyplot as plt


pathshp=pathdat+"/SHP/SHP2/"

#Load shapefiles
gw_sel=gpd.read_file(pathshp+"GWF2.shp") #vector with the wells and locations

#LS GIS information OSM
waterbodies=gpd.read_file(pathshp+"waterbodiesND.shp") #to add in the background map
waterways=gpd.read_file(pathshp+"waterwaysND.shp")
citiesND=gpd.read_file(pathshp+"citiesND2.shp")

#Administrative boundaries
germany_states = gpd.read_file(pathshp+"DEU_adm1.shp")
ND=germany_states[germany_states.NAME_1== "Niedersachsen"]


proj_coor=4647
gw_sel=gw_sel.to_crs(epsg=proj_coor)
waterbodies=waterbodies.to_crs(epsg=proj_coor)
waterways=waterways.to_crs(epsg=proj_coor)

cities=citiesND.to_crs(epsg=proj_coor)
germany_states=germany_states.to_crs(epsg=proj_coor)
ND=ND.to_crs(epsg=proj_coor)

proj_coor=4647
gw_sel=gw_sel.to_crs(epsg=proj_coor)


idw=9700232
gwid=gw_sel.loc[gw_sel.MEST_ID==idw]

#### ms=20
column='r2_2'
fig, axs = plt.subplots(ncols=1, figsize=(8, 5))

gw=gwid.plot(ax=axs, markersize=80,
               marker="v", facecolor=None, zorder=3, color="r")
gw_sel.plot(ax=gw, markersize=8, marker="v", facecolor=None,color="#335c67", alpha=.5, zorder=2)
#wb=waterbodies.plot( ax=axs, alpha=0.5, color='b', linewidth=0.7, zorder=1)
ww=waterways.plot( ax=axs, alpha=0.3, color='b', linewidth=.3,zorder=2)
#gdff=gdf.plot( ax=gw, alpha=0.5, color='r',markersize=12,zorder=2)
NS=ND.boundary.plot( ax=axs, alpha=0.3, edgecolor='k', linewidth=1, 
                    zorder=1)

gw.spines['top'].set_visible(False)
gw.spines['right'].set_visible(False)
gw.spines['bottom'].set_visible(False)
gw.spines['left'].set_visible(False)

gw.get_xaxis().set_ticks([])
gw.get_yaxis().set_ticks([])

plt.tight_layout()
pathfig="D:\FOSTER\Figs/" 
plt.savefig(pathfig+"2loc"+str(idw)+".jpg",bbox_inches="tight",dpi=250)