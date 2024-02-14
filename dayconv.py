"""
Utility for converting between Gregorian and Julian dates.

gd2jd -- convert (string) Gregorian Date into Julian Date
jd2gd -- convert (numerical) Julian Date into Gregorian Date


You can confirm that they are mutually compatible:
   import dayconv
   jd0 = 2451545.0001
   print jd0, dayconv.gd2jd(dayconv.jd2gd(2451545.0001).__str__())
"""
# 2010-06-30 17:44 IJC: Codified for Kelle Cruz's "Python Switchers' Guide"


import matplotlib.dates as dates

def gd2jd(datestr):
    """ Convert a string Gregorian date into a Julian date using Pylab.
        If no time is given (i.e., only a date), then noon is assumed.
        Times given are assumed to be UTC (Greenwich Mean Time).

       EXAMPLES:
            print gd2jd('Aug 11 2007')   ---------------> 2454324.5
            print gd2jd('12:00 PM, January 1, 2000')  --> 2451545.0

       SEE ALSO: jd2gd
       """
    # 2008-08-26 14:03 IJC: Created        
    
    if datestr.__class__==str:
        d = dates.datestr2num(datestr)
        jd = dates.num2julian(d) + 3442850
    else:
        jd = []

    return jd

def jd2gd(juldat):
    """ Convert a numerial Julian date into a Gregorian date using Pylab.
        Timezone returned will be UTC.

       EXAMPLES:
          print jd2gd(2454324.5)  --> 2007-08-12 00:00:00
          print jd2gd(2451545)    --> 2000-01-01 12:00:00

       SEE ALSO: gd2jd"""
    # 2008-08-26 14:03 IJC: Created    
    d = dates.julian2num(juldat)
    gd = dates.num2date(d - 3442850)

    return gd
