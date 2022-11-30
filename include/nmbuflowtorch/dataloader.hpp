#ifndef NMBUFLOWTORCH_DATALOADER_H_
#define NMBUFLOWTORCH_DATALOADER_H_

#include <iostream>
#include <string>
#include <vector>

#include "./definitions.hpp"

// Code taken from https://aleksandarhaber.com/eigen-matrix-library-c-tutorial-saving-and-loading-data-in-from-a-csv-file/
inline Matrix openCSV(std::string filepath)
{
  std::cout << "opening CSV" << std::endl;

  std::vector<float> matrixEntries;
  std::ifstream matrixDataFile(filepath);

  std::string matrixRowString;
  std::string matrixEntry;
  int matrixRowNumber = 0;

  while (std::getline(matrixDataFile, matrixRowString))  // here we read a row by row of matrixDataFile and store every line
                                                         // into the string variable matrixRowString
  {
    std::stringstream matrixRowStringStream(
        matrixRowString);  // convert matrixRowString that is a string to a stream variable.

    while (std::getline(
        matrixRowStringStream, matrixEntry, ','))  // here we read pieces of the stream matrixRowStringStream until every
                                                   // comma, and store the resulting character into the matrixEntry
    {
      matrixEntries.push_back(std::stof(matrixEntry));  // stof casts the string to a float
    }

    matrixRowNumber++;  // update the column numbers
  }

  return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

#endif  // NMBUFLOWTORCH_DATALOADER_H_