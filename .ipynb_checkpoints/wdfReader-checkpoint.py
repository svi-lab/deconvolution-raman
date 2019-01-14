from __future__ import print_function
import struct
import numpy

l_int16 = 2
l_int32 = 4
l_int64 = 8
s_int16 = "<H"
s_int32 = "<I"                  # little edian
s_int64 = "<Q"                  # little edian

l_float = 4
s_float = "<f"
l_double = 8
s_double = "<d"


class wdfReader(object):
    def __init__(self, file_name):
        try:
            self.file_obj = open(file_name, "rb")
        except IOError:
            raise IOError("File {} does noe exist!".format(file_name))
        # Initialize the properties for the wdfReader class
        self.title = ""
        self.username = ""
        self.measurement_type = ""
        self.scan_type = ""
        self.laser_wavenumber = None
        self.count = None
        self.spectral_units = ""
        self.xlist_type = None
        self.xlist_units = ""
        self.ylist_type = None
        self.ylist_units = ""
        self.point_per_spectrum = None
        self.data_origin_count = None
        self.capacity = None
        self.application_name = ""
        self.application_version = [None]*4
        self.xlist_length = None
        self.ylist_length = None
        self.accumulation_count = None
        self.block_info = {}    # each key has value (offset, size)
        # Parse the header section in the wdf file
        try:
            self.parse_header()
        except:
            print("Failed to parse the header of file")
        # Location the data, xlist, ylist and 
        try:
            self.block_info["DATA"] = self.locate_block("DATA")
            self.block_info["XLST"] = self.locate_block("XLST")
            self.block_info["YLST"] = self.locate_block("YLST")
        except:
            print("Failed to get the block information")
        # set xlist and ylist unit and type
        self.xlist_type, self.xlist_units = self.get_xlist_info()
        self.ylist_type, self.ylist_units = self.get_ylist_info()
        # set the data origin, if count is not 0 (for mapping applications)
        if (self.data_origin_count != None) and (self.data_origin_count != 0):
            try:
                self.block_info["ORGN"] = self.locate_block("ORGN")
            except:
                print("Failed to get the block information")
        # TODO
        # self.origin_list_info = self.get_origin_list_info()
        
    def _read_int16(self):
        return struct.unpack(s_int16, self.file_obj.read(l_int16))[0]
    def _read_int32(self):
        return struct.unpack(s_int32, self.file_obj.read(l_int32))[0]
    def _read_int64(self):
        return struct.unpack(s_int64, self.file_obj.read(l_int64))[0]
    def _read_float(self):
        return struct.unpack(s_float, self.file_obj.read(l_float))[0]
    def _read_double(self):
        return struct.unpack(s_double, self.file_obj.read(l_double))[0]
    def _read_utf8(self, size):
        # TODO: strip the blanks
        return self.file_obj.read(size).decode("utf8")

    # The method for reading the info in the file header 
    def parse_header(self):
        self.file_obj.seek(0)   # return to the head
        block_ID = self.file_obj.read(4).decode("ascii")  # Must make the conversion under python3
        block_UID = self._read_int32()
        block_len = self._read_int64()
        if (block_ID != "WDF1") or (block_UID != 0 and block_UID != 1) \
           or (block_len != 512):
            raise ValueError("The wdf file format is incorrect!")
        #block_uuid = self._read_int32(4) # unique file identifier - never changed once allocated
        # The keys from the header
        self.file_obj.seek(60)
        self.point_per_spectrum = self._read_int32() # this is the number of differents wavelengths at which each spectra was recorded
        self.capacity = self._read_int64() # this is the number of points at which we recorded the spectra (points in a map scan, for exemple)
        self.count = self._read_int64() # this seems to be the same as above?
        self.accumulation_count = self._read_int32()
        self.ylist_length = self._read_int32()
        self.xlist_length = self._read_int32() # this seems to be the same as .point_per_spectrum ?
        self.data_origin_count = self._read_int32() # no idea what this is
        self.application_name = self._read_utf8(24) # Wire + bunch of blanks
        for i in range(4):
            self.application_version[i] = self._read_int16()
        # TODO: change the types to string
        self.scan_type = self._read_int32() # it seems to give 7 for the map scan
        self.measurement_type = self._read_int32() # it gives 3 for my map scan
        # For the units
        # TODO: change to string
        self.file_obj.seek(152)
        self.spectral_units = self._read_int32() # this gives 6 form my file, so it should correspond to cm-1 ?
        self.laser_wavenumber = self._read_float() # doing (10 000 000 / laser_wavenumber) should give you the wavelength in nm
        # Username and title
        self.file_obj.seek(208)
        self.username = self._read_utf8(32) # windows session login (one should strip the blanks here)
        self.title = self._read_utf8(160) # StreamHR image acquisition (strip the blanks)

    # locate the data block offset with the corresponding block name
    def locate_block(self, block_name):
        if (block_name in self.block_info) and (self.block_info[block_name] is not None):
            return self.block_info[block_name]
        else:
            # find the block by increment in block size
            # exhaustive but no need to worry
			# so it basically starts from the beggining and goes block by block, reading the block name and size, then moving the cursor at the end of the block, then reading the name of the next block, than it's size etc.
            curr_name = None
            curr_pos = 0
            next_pos = curr_pos
            self.file_obj.seek(curr_pos)
            while (curr_name != block_name) and (curr_name != ""):
                curr_pos = next_pos
                curr_name = self.file_obj.read(4).decode("ascii")  # Always a 4-str block name
                uid = self._read_int32()
                size = self._read_int64()
                next_pos += size
                self.file_obj.seek(next_pos)
                print(curr_name, curr_pos, uid, size)
            # found the id
            if curr_name == block_name:
                return (curr_pos, size)
            else:
                raise ValueError("The block with name {} is not found!".format(block_name))   

    
    
    
    def get_map_area(self):
        pos, size = self.locate_block('WMAP')
        map_type = self._read_int32()
        offset = 8
        self.file_obj.seek(pos + offset)
        
        mapa = numpy.round(numpy.fromfile(self.file_obj, dtype="float32", count=6),2) # first three are the origin coordinates, the next three are stepsizes (?)
#        ox = self._read_float()
#        oy = self._read_float()
#        oz = self._read_float()
#        dx = self._read_float()
#        dy = self._read_float()
#        dz = self._read_float()
        rx = self._read_int32()
        ry = self._read_int32()
        rz = self._read_int32()
        linefocus_size = self._read_int32()
        return (map_type, mapa, rx, ry, rz, linefocus_size) #ox, oy, oz, dx, dy, dz
    
    
    def get_ctime(self):
        pos, size = self.locate_block("CreationTime")
        t1 = self._read_int32()
        t2 = self._read_int32()
        t3 = self._read_int32()
        t4 = self._read_int32()
        t5 = self._read_int32()
        t6 = self._read_int32()        
        return (t1,t2,t3,t4,t5,t6)    
    def get_etime(self):
        pos, size = self.locate_block("ETA")
        et1 = self._read_int32()
        et2 = self._read_int32()
        et3 = self._read_int32()
        et4 = self._read_int32()
        et5 = self._read_int32()
        et6 = self._read_int32()        
        return (et1,et2,et3,et4,et5,et6)    
        
    # get the xlist info
    def get_xlist_info(self):
        pos, size = self.locate_block("XLST")
        offset = 16
        self.file_obj.seek(pos + offset)
        #TODO: strings
        data_type = self._read_int32()
        data_unit = self._read_int32()
        return (data_type, data_unit)

    # get the ylist info
    def get_ylist_info(self):
        pos, size = self.locate_block("YLST")
        offset = 16
        self.file_obj.seek(pos + offset)
        #TODO: strings
        data_type = self._read_int32()
        data_unit = self._read_int32()
        return (data_type, data_unit)

    # TODO: get the origin list info
    # def get_origin_list_info(self):
    #     return (None, None)

    """
    Important parts for data retrieval
    """
    def get_xdata(self):
        pos = self.locate_block("XLST")[0]
        offset = 24
        self.file_obj.seek(pos + offset)
        size = self.xlist_length
        self.file_obj.seek(pos + offset)
        x_data = numpy.fromfile(self.file_obj, dtype="float32", count=size)
        return x_data
    
    def get_ydata(self):
        pos = self.locate_block("YLST")[0]
        offset = 24
        self.file_obj.seek(pos + offset)
        size = self.ylist_length
        self.file_obj.seek(pos + offset)
        y_data = numpy.fromfile(self.file_obj, dtype="float32", count=size)
        return y_data

    def get_spectra(self, start=0, end=-1):
        if end == -1:           # take all spectra
            end = self.count-1
        if (start not in range(self.count)) or (end not in range(self.count)):
            raise ValueError("Wrong start and end indices of spectra!")
        if start > end:
            raise ValueError("Start cannot be larger than end!")

        pos_start = self.locate_block("DATA")[0] + 16 + l_float*start*self.point_per_spectrum
        n_row = end - start + 1
        self.file_obj.seek(pos_start)
        spectra_data = numpy.fromfile(self.file_obj, dtype="float32", count=n_row*self.point_per_spectrum)
        if len(spectra_data.shape) == 1:
            # The spectra is only 1D array
            spectra_data = spectra_data.reshape(n_row, -1)
            return spectra_data
        else:
            # Make 2D array
            spectra_data = spectra_data.reshape(n_row, spectra_data.size // n_row)
            return spectra_data
    
"""
when using this, you should first run the script
than get the object by running kiko = wdfReader(./filename)
then perhaps spektar = wdfReader.get_spectra(kiko)
then you can reshape the spectre by running : novi = spektar.reshape((-1, len(xdata)))
where xdata was obteined by xdata = wdfReader.get_xdata(kiko)

wdfReader.locate_block(kiko,'WMAP') works as well

script reads the block name as 4 bytes ascii, than 4 bytes for the uid (32bit integer), 
then comes the size of the block as unsigned 64bit integer

"""
