import numpy as np
import pandas as pd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

def to_excel(df):
    """ Function to 
        download  
        .xlsx files """

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    writer.save()
    processed_data = output.getvalue()
    return processed_data