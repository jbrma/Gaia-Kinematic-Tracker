from astropy.coordinates import SkyCoord
import astropy.units as u
  
# 11:24:39.0           1  rkp+11     -59:16:19
c = SkyCoord('11:24:39.0 ', '-59:16:19', unit=(u.hourangle, u.deg))

print(f"RA Decimal: {c.ra.degree}")
print(f"DEC Decimal: {c.dec.degree}")