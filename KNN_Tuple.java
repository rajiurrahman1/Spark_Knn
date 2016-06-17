// Author: Rajiur Rahman ( rajiurrahman.bd@gmail.com )
// Department of Computer Science, Wayne State University

// For installation and running of the codes, please find the appropriate tutorial from the following link
//      http://dmkd.cs.wayne.edu/TUTORIAL/Bigdata/Codes/


package org.sparkexample;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;

import scala.Tuple2;
import scala.Tuple3;
import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

public class Tuple{
    private static final Pattern SPACE = Pattern.compile(" ");
    static Vector parseVector(String line) {
        String[] splits = SPACE.split(line);
        double[] data = new double[splits.length];
        int i = 0;
        for (String s : splits) {
            data[i] = Double.parseDouble(s);
            i++;
        }
        splits = null;
        return new Vector(data);
    }

    public static class ReadData implements Function<String, Vector>{
        public Vector call(String s){
            return parseVector(s);
        }
    }

    private static void printTestData(double[][] testData) {
        for(int i=0; i<testData.length; i++){
            for(int j=0; j<testData[0].length; i++){
                System.out.print(testData[i][j]);
            }
            System.out.print("\n");
        }
    }

    public static double[][] readTestData(String filename, int numRows) throws FileNotFoundException, IOException {
        double[][] matrix = {{1}, {2}};
        File inFile = new File(filename);
        Scanner in = new Scanner(inFile);
        int intLength = 0;
        String[] length = in.nextLine().trim().split("\\s+");
        for (int i = 0; i < length.length; i++) {
            intLength++;
        }
        in.close();

        int lineCount = 0;
        matrix = new double[numRows][intLength];
        in = new Scanner(inFile);
        while (in.hasNextLine()) {
            String[] currentLine = in.nextLine().trim().split("\\s+");
            for (int i = 0; i < currentLine.length; i++) {
                matrix[lineCount][i] = Double.parseDouble(currentLine[i]);
            }
            lineCount++;
        }
        return matrix;
    }

    public static void main (String[] args) throws FileNotFoundException, IOException{
        long startTime = System.currentTimeMillis();
        SparkConf sparkConf = new SparkConf().setAppName("TestKMeans");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        String trainFileName = args[0];
        final int numTestInstances = Integer.parseInt((args[1]));
        final int dataDimension =Integer.parseInt((args[2]));
        final int k = Integer.parseInt((args[3]));
        String testFileName = args[4];

        JavaRDD<String> trainLines = sc.textFile(trainFileName);
        final JavaRDD<Vector> trainData = trainLines.map(new ReadData());
        Long trainDataCount = trainData.count();long currentTime1 = System.currentTimeMillis();
        System.out.println("\nTraining data read completed\nTime taken: "+(currentTime1-startTime)+"\n\n");

        final double[][] testData = readTestData(testFileName, numTestInstances);

        JavaRDD<Tuple3<Integer, Long, Double>> dTuple3 = trainData.zipWithIndex().flatMap(
                new FlatMapFunction<Tuple2<Vector, Long>, Tuple3<Integer, Long, Double>>() {
                    @Override
                    public Iterable<Tuple3<Integer, Long, Double>> call(Tuple2<Vector, Long> trainTuple) throws Exception {
                        ArrayList<Tuple3<Integer, Long, Double>> ret = new ArrayList<Tuple3<Integer, Long, Double>>();
                        for (int i = 0; i < testData.length; i++) {
                            double d = calculateDistance(testData, i, trainTuple._1());
                            ret.add(new Tuple3<Integer, Long, Double>(i, trainTuple._2(), d));
                        }

                        return ret;
                    }

                    private double calculateDistance(double[][] testData, int i, Vector trainVector) {
                        double d = (double) 0.0;
                        for (int j = 0; j < dataDimension - 1; j++) {
                            d += Math.pow((testData[i][j] - trainVector.elements()[j]), 2);
                        }
                        return Math.sqrt(d);
                    }
                }
        );
        Long dTuple3Count = dTuple3.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\nTuple3 generation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");


        JavaPairRDD<Integer, Tuple2<Long, Double> > dPair = dTuple3.mapToPair(
                new PairFunction<Tuple3<Integer, Long, Double>, Integer, Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Integer, Tuple2<Long, Double>> call(Tuple3<Integer, Long, Double> t3) throws Exception {
                        Tuple2<Long, Double> t2 = new Tuple2<Long, Double>(t3._2(), t3._3());
                        return new Tuple2<Integer, Tuple2<Long, Double>>(t3._1(), t2);
                    }
                }
        );

        Long dPairCount = dPair.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\ndistance pair generation completed after mapToPair of dTuple3\nTime taken: "+(currentTime1-startTime)+"\n\n");

        JavaRDD<Long[]> nearestNeighborIndices = dPair.groupByKey().map(
                new Function<Tuple2<Integer, Iterable<Tuple2<Long, Double>>>, Long[]>() {
                    @Override
                    public Long[] call(Tuple2<Integer, Iterable<Tuple2<Long, Double>>> iterableT2) throws Exception {
                        Long[] nnIndices = new Long[k + 1];
                        ArrayList<Tuple2<Long, Double>> tempList = new ArrayList<Tuple2<Long, Double>>();
                        for (int i = 0; i < k; i++) {
                            tempList.add(new Tuple2<Long, Double>(1L, 999999.0));
                        }
                        for (Tuple2<Long, Double> t2 : iterableT2._2()) {
                            //if t2-distance is less than any of the item of tempList, then replace t2 with the max item of tempList
                            int checkIndex = checkNearestNeighbor(tempList, t2, k);
                            if (checkIndex != -1) {
                                tempList.set(checkIndex, t2);
                            }
                        }
                        for (int i = 0; i < k; i++) {
                            nnIndices[i] = tempList.get(i)._1();
                        }
                        nnIndices[k] = Long.valueOf(iterableT2._1());  // the last value of this array will hold the TestRow number
                        return nnIndices;
                    }

                    private int checkNearestNeighbor(ArrayList<Tuple2<Long, Double>> tempList, Tuple2<Long, Double> t2, int k) {
                        int flag = 0;
                        int ret = -1;  // if current instance doesn't belong to nearest neighbor List, return -1
                        // we will find the maximum distance value and it's index in the tempList
                        // if current t2->distance is less than any of the values in tempList, we will send the index of highest distance
                        double tempMaxValue = tempList.get(0)._2();
                        int tempMaxIndex = 0;
                        for (int i = 0; i < k; i++) {
                            if (t2._2() < tempList.get(i)._2()) {
                                flag = 1;
                            }
                            if (tempList.get(i)._2() > tempMaxValue) {      // check the max value and save index of maxValue
                                tempMaxIndex = i;
                                tempMaxValue = tempList.get(i)._2();
                            }
                        }
                        if (flag == 1) {
                            ret = tempMaxIndex;
                        }
                        return ret;
                    }
                }
        );
        Long nnIndicesCount = nearestNeighborIndices.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\nNearest neighbor inices calculation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");



        JavaRDD<Double> trainingClassLabels = trainData.map(
                new Function<Vector, Double>() {
                    public Double call(Vector vector) throws Exception {
                        int dataDimension = (int) vector.length();
                        return vector.elements()[dataDimension - 1];
                    }
                }
        );
        Long lablesCount = trainingClassLabels.count(); currentTime1 = System.currentTimeMillis();
        System.out.println("\nclass label generation completed\nTime taken: "+(currentTime1-startTime)+"\n\n");


        final List<Double>trainingClassLabelsList = trainingClassLabels.collect();
        JavaRDD<Integer[]> predictedLabels = nearestNeighborIndices.map(    //first element will be testRowNum, next is PredictedClassLabel
                new Function<Long[], Integer[]>() {
                    @Override
                    public Integer[] call(Long[] indices) throws Exception {
                        Integer[] ret = new Integer[2];
                        int testRowNum = indices.length;
                        ret[0] = (int) (long) indices[testRowNum - 1];
                        ret[1] = predictClassLabels(indices, trainingClassLabelsList);
                        return ret;
                    }

                    private int predictClassLabels(Long[] indices, List<Double> trainingClassLabelsList) {
                        int zeroSum = 0;
                        int oneSum = 0;
                        for (int i = 0; i < indices.length - 1; i++) {            // the last element of indices array is the testRowNum
                            int currentIndex = (int) (long) indices[i];
                            if (trainingClassLabelsList.get(currentIndex) == 0) {
                                zeroSum++;
                            } else {
                                oneSum++;
                            }
                        }
                        if (oneSum >= zeroSum) {
                            return 1;
                        } else {
                            return 0;
                        }
                    }
                }
        );


        Random randomGenerator = new Random();
        int randomInt = randomGenerator.nextInt(100000);
        String outFileName = "labels_"+Integer.toString(k)+"_"+Integer.toString(dataDimension)+"_"+Integer.toString(randomInt)+".txt";
        predictedLabels.saveAsTextFile(outFileName);
        System.out.println("Predicted Labels output written to " + outFileName + "\n\n");

        sc.stop();
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("\n\n\n\n\nEnd of Spark KNN CLassification\nTotal time taken: " + totalTime + " ms\n\n\n");

    }

}
