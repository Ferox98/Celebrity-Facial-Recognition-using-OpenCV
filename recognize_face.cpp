#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <iostream>
#include <vector>
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>

using namespace std;

using namespace boost::filesystem;

using namespace cv;

typedef tuple<vector<Mat>, vector<int>> dataset;

// define a dictionary that maps celebrities to their respective ids
map<int, string> dict;

vector<directory_entry> readDirectory(string path) {
  vector<directory_entry> v;
  // save all the files within the directory
  copy(directory_iterator(path), directory_iterator(), back_inserter(v));
  return v;
}

tuple<dataset, dataset> trainTestSplit(dataset d) {
  vector<Mat> images = get<0>(d);
  vector<int> labels = get<1>(d);
  // assign 75% of the entire dataset to the training and the remaining 25% to the testing datasets
  int i = 0;

  vector<Mat> training_images, test_images;
  vector<int> training_labels, test_labels;

  for(i; i < (int) (images.size() * 0.75); i++) {
    training_images.push_back(images[i]);
    training_labels.push_back(labels[i]);
  }
  for(i; i < images.size(); i++) {
    test_images.push_back(images[i]);
    test_labels.push_back(labels[i]);
  }

  dataset training_samples = make_tuple(training_images, training_labels);
  dataset test_samples = make_tuple(test_images, test_labels);
  return make_tuple(training_samples, test_samples);
}

tuple<dataset, dataset> saveImages(vector<directory_entry>& paths) {
    // define the vectors of the reference images and their labels
    dataset training_samples, test_samples;
    int celeb_id = 0;

    for (vector<directory_entry>::const_iterator it = paths.begin(); it != paths.end(); it++) {
      // get the celebrity's name
      string path = (*it).path().string();
      string celeb = path.substr(10);
      dict[celeb_id] = celeb;

      vector<directory_entry> image_dir = readDirectory(path);
      vector<Mat> images;
      vector<int> labels;
      for (vector<directory_entry>::const_iterator it2 = image_dir.begin(); it2 != image_dir.end(); it2++) {
        string img_path = (*it2).path().string();
        images.push_back(imread(img_path, IMREAD_GRAYSCALE));
        labels.push_back(celeb_id);
      }
      // split the dataset into training and testing testing_samples
      dataset d = make_tuple(images, labels);
      tuple<dataset, dataset> samples = trainTestSplit(d);
      dataset training_ds = get<0>(samples);
      dataset test_ds = get<1>(samples);
      get<0>(training_samples).insert(end(get<0>(training_samples)), get<0>(training_ds).begin(), get<0>(training_ds).end());
      get<1>(training_samples).insert(end(get<1>(training_samples)), get<1>(training_ds).begin(), get<1>(training_ds).end());
      get<0>(test_samples).insert(end(get<0>(test_samples)), get<0>(test_ds).begin(), get<0>(test_ds).end());
      get<1>(test_samples).insert(end(get<1>(test_samples)), get<1>(test_ds).begin(), get<1>(test_ds).end());

      celeb_id++;
    }
    return make_tuple(training_samples, test_samples);
}

int main() {
  string path = "PINS";
  cout << "Reading image directory" << endl;
  vector<directory_entry> paths = readDirectory(path);
  cout << "Fetching images" << endl;
  tuple<dataset, dataset> samples = saveImages(paths);
  dataset training_samples = get<0>(samples), test_samples = get<1>(samples);

  cout << "Training recognizer" << endl;
  Ptr<face::FaceRecognizer> recognizer = face::LBPHFaceRecognizer::create(1, // radius of LBP Pattern
                                                                        8, // No. of neignboring pixels to consider
                                                                        8, 8, // Grid size
                                                                        200. );// minimum distance to nearest neighbor

  vector<Mat> ref_images = get<0>(training_samples);
  vector<int> ref_labels = get<1>(training_samples);
  recognizer->train(ref_images, ref_labels);
  cout << "Training complete" << endl;
  vector<Mat> test_images = get<0>(test_samples);
  vector<int> test_labels = get<1>(test_samples);
  int predicted_label = -1;
  double confidence = 0.0;
  recognizer->predict(test_images[130], predicted_label, confidence);
  cout << "Image label = " << dict[test_labels[130]] << " , Predicted label = " << dict[test_labels[predicted_label]] << " , confidence = " << confidence << endl;
}
