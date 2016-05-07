// This program aims to add an extra image list file to an already existing LEVELDB/LMDB file
// in a random order. What we do is:
// 1. We first open the existing LMDB/LEVELDB file.
// 2. We read the key-value pair one by one.
// 3. We randomly insert the image list file into the existing LMDB/LEVELDB file
// 4. We then write the destination file with the refined the key-value list.
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

//using namespace leveldb;
using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool( gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool( shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string( existing_backend, "lmdb",
        "The backend {lmdb, leveldb} for the existing file");
DEFINE_string( destination_backend, "lmdb",
        "The backend {lmdb, leveldb} for the destination file");
DEFINE_int32( resize_width, 0, "Width images are resized to");
DEFINE_int32( resize_height, 0, "Height images are resized to");
DEFINE_bool( check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool( encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string( encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

DEFINE_string( img_list, "",
    "the image list file that should be added to the destination file.");
DEFINE_string( db_save_name, "",
    "the new DB save file name");
DEFINE_string( db_existing_file, "",
    "the directory that the db existing file lies in");
DEFINE_string( exist_img_list, "",
    "the existing the image list file, which is used to calculate the stored image number.");


int main(int argc, char** argv) {

#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 1 ) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset_incremental_add");
    return 1;
  }

  CPUTimer total_time;
  total_time.Start();
  
  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const std::string existing_backend = FLAGS_existing_backend;
  const std::string destination_backend = FLAGS_destination_backend;
  const std::string db_save_name = FLAGS_db_save_name;
  const std::string img_list = FLAGS_img_list;
  CHECK( img_list.size() > 0 ) << "the input img_list must contain at least one image name and its corresponding label name.";
  
  const std::string exist_img_list = FLAGS_exist_img_list;
  const string db_existing_file = FLAGS_db_existing_file;
  
  std::vector<int> img_total_label;// 0 means we store img, 1 means we store the existing db file

  CPUTimer calcu_line_num_time;
  calcu_line_num_time.Start();
  //calculating the existing image number stored in DB file;
  int exist_img_num = 0;
  if( exist_img_list.size() > 0 ){
    std::ifstream exist_img_list_file( exist_img_list.c_str() );
    int label_tmp = 0;
    std::string img_name_tmp;
    while( exist_img_list_file >> img_name_tmp >> label_tmp ){
      exist_img_num++;
      img_total_label.push_back( 1 );
    }
    exist_img_list_file.close();
  }else{
    scoped_ptr<db::DB> db_exist( db::GetDB(existing_backend) );
    db_exist -> Open( db_existing_file, db::READ );
    //scoped_ptr<db::Transaction> txn_exist( db_exist->NewTransaction() );
    scoped_ptr<db::Cursor> cursor_exist( db_exist -> NewCursor() );
    cursor_exist -> SeekToFirst();
    while( cursor_exist -> valid() ){
      exist_img_num++;
      cursor_exist -> Next();
      img_total_label.push_back( 1 );
    }
    db_exist -> Close();
  }
  CHECK( exist_img_num > 0 ) << "the existing image num saved in DB should be larger than 0, please check!";
  calcu_line_num_time.Stop();
  //calculating the new adding image number and further store image name and its corresponding label in img_lines.
  std::ifstream infile( img_list.c_str() );
  std::vector< std::pair<std::string, int> > img_lines;
  std::string filename;
  int label;
  while ( infile >> filename >> label ) {
    img_lines.push_back( std::make_pair( filename, label ) );
    img_total_label.push_back( 0 );
  }
  CHECK( img_lines.size() > 0 ) << "the newly added image number should be larger than 0, please check!";
  LOG(INFO) << "We have to add a total of " << img_lines.size() << " images.";
  
   
  //Randomly insert the newly added images into the existing DB file;
  //Step1: randomly shuffle the img_total_label;
  CHECK( img_total_label.size() > 0 ) << "error! the img_total_label.size() must be larger than 0 ";
  if( FLAGS_shuffle == true ){
    LOG( INFO ) << "Shuffling data ...";
    shuffle( img_total_label.begin(), img_total_label.end() );
  }
  //Step2: store the image/DB file one by one according to img_total_label
  //Note: if img_total_label[i] == 0, it means we should store newly added image file;
  //if img_total_label[i] == 1, it means we should store the existing DB file;
  scoped_ptr<db::DB> db_save( db::GetDB( destination_backend ) );
  db_save->Open( db_save_name, db::NEW );
  scoped_ptr<db::Transaction> txn_save( db_save->NewTransaction() );


  scoped_ptr<db::DB> db_exist( db::GetDB( existing_backend ) );
  //RepairDB( db_existing_file, db::READ );
  db_exist -> Open( db_existing_file, db::READ );
  scoped_ptr<db::Transaction> txn_exist( db_exist->NewTransaction() );
  scoped_ptr<db::Cursor> cursor_exist( db_exist -> NewCursor() );
  cursor_exist -> SeekToFirst();
  

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  //scoped_ptr<db::DB> db_img( db::GetDB( destination_backend ) );
  // get the handler
  //scoped_ptr<db::Transaction> txn_img( db_img->NewTransaction() );
  //scoped_ptr<db::Cursor> cursor_ptr( db->NewCursor() ); 

  // Storing to db
  //std::string root_folder( argv[1] );
  Datum datum;
  //int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  int img_tag = 0;
  for( int i = 0; i < img_total_label.size(); ++i ){
    if( img_total_label[i] == 0 ){
      bool status = false;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
        // Guess the encoding type from the file name
        std::string fn = img_lines[ img_tag ].first;
        size_t p = fn.rfind('.');
        if ( p == fn.npos )
          LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
          enc = fn.substr(p);
          std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum( img_lines[ img_tag ].first,
        img_lines[ img_tag ].second, resize_height, resize_width, is_color, enc, &datum );
        if( status == false ) continue;
        if( check_size ) {
          if (!data_size_initialized) {
            data_size = datum.channels() * datum.height() * datum.width();
            data_size_initialized = true;
          } else {
            const std::string& data = datum.data();
            CHECK_EQ( data.size(), data_size ) << "Incorrect data field size "
                << data.size();
          }
        }
        std::string key_str = caffe::format_int( i, 8 ) + "_" + img_lines[ img_tag ].first;
        std::string val_str;
        CHECK( datum.SerializeToString( &val_str ) );
        txn_save -> Put( key_str, val_str );
        img_tag++;
     }
     
     //otherwise, we save DB file accordingly;
     if( img_total_label[i] ==  1 ){
        if( cursor_exist -> valid() ){
           std::string key_str_old = cursor_exist -> key();
           std::string key_str_strip = key_str_old.substr( key_str_old.find_first_of( "_", 0 ) + 1,
                                                    key_str_old.size() - 1 );
           //LOG( INFO ) << "key_str_strip = \n" << key_str_strip << std::endl;
           std::string key_str_save = caffe::format_int( i, 8 ) + "_" + key_str_strip;
           std::string val_str = cursor_exist -> value();
           txn_save -> Put( key_str_save, val_str );
           cursor_exist -> Next();     
        }     
     }
     if( i % 1000 == 0 ){
       txn_save -> Commit();
       txn_save.reset( db_save -> NewTransaction() );
       LOG(INFO) << "Processed " << i << " files.";
     }
  }
  //db_exist -> Close();
  if ( img_total_label.size()%1000 != 0 ) {
    txn_save -> Commit();
    LOG(INFO) << "Write " << img_total_label.size() << " files.";
  }
  total_time.Stop();
  LOG( INFO ) << "the total time used is: " << float(total_time.MilliSeconds())/60000.0 << " mins.";
  LOG( INFO ) << "the calculate lines time used is " << float(calcu_line_num_time.MilliSeconds())/60000.0 << " mins.";  
  //db_exist -> Close();
  //db_save -> Close(); 
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
