# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:21:01 2024
Class to hold all kind of tools

@author: kristina
"""



from datetime import datetime



class tools:       
    
   
    
    def dt(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - '

    # Define a function called date_diff_in_Seconds which calculates the difference in seconds between two datetime objects
    def dateDifSeconds(self):
        # Calculate the time difference between dt2 and dt1
        startDate = datetime.strptime('2024-06-17 01:00:00', '%Y-%m-%d %H:%M:%S')
        timedelta = datetime.now() - startDate
        # Return the total time difference in seconds
        return timedelta.days * 24 * 3600 + timedelta.seconds

    # Local to ISO 8601 without microsecond:
    def getDateInIso8601(self):
        d = datetime.now().isoformat(timespec='seconds', sep=' ')
        d = d.replace(":","")
        return d
              
    
    
    
   