import Text.Printf
import Control.Arrow ((***))

-- The haversine of an angle.
haversine :: Float -> Float
haversine = (^ 2) . sin . (/ 2)

-- The approximate distance, in kilometers, between two points on Earth.
-- The latitude and longtitude are assumed to be in degrees.
earthDist :: (Float, Float) -> (Float, Float) -> Float
earthDist = distDeg 6371
  where
    distDeg radius p1 p2 = distRad radius (deg2rad p1) (deg2rad p2)
    distRad radius (lat1, lng1) (lat2, lng2) =
      (2 * radius) *
      asin
        (min
           1.0
           (sqrt $
            haversine (lat2 - lat1) +
            ((cos lat1 * cos lat2) * haversine (lng2 - lng1))))
    deg2rad = d2r *** d2r
      where
        d2r = (/ 180) . (pi *)

main :: IO ()
main =
  printf
    "The distance between BNA and LAX is about %0.f km.\n"
    (earthDist bna lax)
  where
    bna = (36.12, -86.67)
    lax = (33.94, -118.40)
