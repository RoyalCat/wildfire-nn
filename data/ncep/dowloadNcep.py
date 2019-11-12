import os
import urllib.request
import asyncio
import wget

yearStart = 2012
yearStop = 2017

urls_to_download = []

siteUrl = "https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep/"

for year in range(yearStart, yearStop):
    print("\nYear:" + str(year))
    wget.download(siteUrl + "air." + str(year) + ".nc")
    wget.download(siteUrl + "rhum." + str(year) + ".nc")
    wget.download(siteUrl + "uwnd." + str(year) + ".nc")
#for url in urls_to_download:
#    wget.download(url)