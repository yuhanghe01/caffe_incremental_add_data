# Caffe C++ implementation for adding new image data to existing LEVELDB/LMDB database
One excruciating experience when working with Caffe is to create LEVELDB/LMDB database. Traditionally, if you have to add new image data to existing databse, current method requires to create the whole database again, which is time consuming. For example, during my 
