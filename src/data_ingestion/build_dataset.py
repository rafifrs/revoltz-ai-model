import pandas as pd
from pathlib import Path
import zipfile
import scipy.io
import io
import numpy as np

def build_nasa_pcoe_dataset(input_dir: Path) -> pd.DataFrame:
    """
    Build a dataset specifically for NASA PCoE data from .mat files inside .zip.
    """
    all_data = []
    
    for zip_file in input_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.mat'):
                    try:
                        with z.open(filename) as f:
                            mat = scipy.io.loadmat(io.BytesIO(f.read()))
                            
                            # Asumsi standar penamaan PCoE: nama variabel sama dengan nama file
                            battery_name = filename.split('/')[-1].split('.')[0]
                            
                            if battery_name not in mat:
                                print(f"Variable {battery_name} tidak ditemukan dalam {filename}")
                                continue
                                
                            # Struktur standar NASA PCoE: mat[battery_name][0][0]['cycle'][0]
                            cycles = mat[battery_name][0][0]['cycle'][0]
                            
                            extracted_data = []
                            discharge_cycle_count = 1
                            
                            # Jika tidak ada nominal capacity (sebagai denominator untuk SoH), kita akan menggunakan
                            # kapasitas maksimum yang dicatat di siklus discharge awal.
                            max_capacity = 0 
                            temp_extracted = []
                            
                            for cycle in cycles:
                                op_type = cycle['type'][0]
                                
                                # Evaluasi Degradasi SoH fokus pada cycle 'discharge'
                                if op_type == 'discharge':
                                    data_dict = cycle['data'][0][0]
                                    
                                    try:
                                        # Kapasitas yang terekam pada cycle discharge tersebut
                                        capacity_ah = data_dict['Capacity'][0][0]
                                        if capacity_ah > max_capacity:
                                            max_capacity = capacity_ah
                                    except (ValueError, KeyError, IndexError):
                                        capacity_ah = np.nan
                                    
                                    try:
                                        # Rata-rata dari time-series pengukuran suhu dan voltase
                                        temp_array = data_dict['Temperature_measured'][0]
                                        temp_c = temp_array.mean() if len(temp_array) > 0 else np.nan
                                        
                                        voltage_array = data_dict['Voltage_measured'][0]
                                        ocv_v = voltage_array.mean() if len(voltage_array) > 0 else np.nan
                                    except (ValueError, KeyError, IndexError):
                                        temp_c = np.nan
                                        ocv_v = np.nan

                                    temp_extracted.append({
                                        'battery_id': battery_name,
                                        'source': 'nasa_pcoe',
                                        'cycle_count': discharge_cycle_count,
                                        'capacity_ah': capacity_ah,
                                        'temperature_c': temp_c,
                                        'ocv_v': ocv_v,
                                        'age_days': 0.0, # Dummy fallback, as .mat doesn't strictly have chronological age
                                        'chemistry': 'li-ion'
                                    })
                                    discharge_cycle_count += 1
                            
                            # Hitung Target SoH (State of Health)
                            # Best practice estimasi label SoH: (Current Capacity / Nominal/Max Capacity)
                            if max_capacity > 0:
                                for record in temp_extracted:
                                    # Normalisasi agar SoH berada di rentang yang logis
                                    calculated_soh = record['capacity_ah'] / max_capacity
                                    record['soh'] = float(np.clip(calculated_soh, 0.0, 1.0))
                            else:
                                for record in temp_extracted:
                                    record['soh'] = np.nan

                            if temp_extracted:
                                all_data.append(pd.DataFrame(temp_extracted))
                            
                    except Exception as e:
                        print(f"Error memproses {filename}: {str(e)}")

    if not all_data:
         raise ValueError(f"Tidak ada file .mat yang bisa diproses dalam file ZIP di {input_dir}")

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

def split_dataset_by_battery(df: pd.DataFrame):
    battery_ids = df['battery_id'].unique()
    train_ids = battery_ids[:int(0.7 * len(battery_ids))]
    val_ids = battery_ids[int(0.7 * len(battery_ids)):int(0.85 * len(battery_ids))]
    test_ids = battery_ids[int(0.85 * len(battery_ids)):]  # Remaining 15%

    train_df = df[df['battery_id'].isin(train_ids)]
    val_df = df[df['battery_id'].isin(val_ids)]
    test_df = df[df['battery_id'].isin(test_ids)]

    return train_df, val_df, test_df