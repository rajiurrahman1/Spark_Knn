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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.regex.Pattern;

public class Broadcast{
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

        String testFileName = args[0];
        String trainFileName = args[1];
        final int dataDimension = Integer.parseInt((args[2]));
        final int numTrainInstances = Integer.parseInt((args[3]));
        final int k = Integer.parseInt((args[4]));

        System.out.println("\n\n\n\n\n\nHello World from Broadcast!\n\n\n\n\n\n");
        JavaRDD<String> testLines = sc.textFile(testFileName);
        final JavaRDD<Vector> testData = testLines.map(new ReadData());
        long trainDataCount = testData.count();
        System.out.println("\n\n\nTest data read completed\n\n\n");

        final double[][] trainData = readTestData(trainFileName, numTrainInstances );
        System.out.println("\n\n\n\nTrain data read completed. Total test rows - "+ trainData.length + "\n\n\n\n");


        //broadcast the test data into the worker nodes.
        final org.apache.spark.broadcast.Broadcast<double[][]> broadcastedTrainData = sc.broadcast(trainData);

        System.out.println("\n\nBroadcasting training data - Done \n\n");
        //zipping train data with index i.e. adding a row number for each row
        //the resultant will be a tuple2 variable where _1() will be the vector and _2() will be row number
        //The format of the zipped variable will be following
        //[((1.0, 1.0, 1.0, 1.0),0), ((2.0, 2.0, 2.0, 0.0),1)]
        JavaPairRDD<Vector, Long> testRDDZipped = testData.zipWithIndex();
        long tempCount = testRDDZipped.count();
        System.out.println("\n\nZip test data with index - Done \n\n");

        //now calculate the distance from each trainRow to each testRow
        //this RDD will contain the row number of trian data, and a Vector of size |testData|
        //the values of the vector will be the distances
        JavaPairRDD<Vector,Long> distances = testRDDZipped.mapToPair(
                new PairFunction<Tuple2<Vector, Long>, Vector, Long>() {
                    @Override
                    public Tuple2<Vector, Long> call(Tuple2<Vector, Long> vectorLongT2) throws Exception {
                        double[][] trainData1 = broadcastedTrainData.getValue(); //the test data is retrieved from broadcast
                        Vector testVector = vectorLongT2._1();
//                        Long trainRowNumber = vectorLongT2._2();
                        double[] distanceArray = new double[trainData1.length];
                        Arrays.fill(distanceArray, 0.0);

                        for (int i = 0; i < trainData1.length; i++) {
                            distanceArray[i] = calculateDistanceBetweenRows(testVector, trainData1, i);
                        }
                        Vector vDistance = new Vector(distanceArray);
                        distanceArray = null;
                        return new Tuple2<Vector, Long>(vDistance, vectorLongT2._2());
                    }

                    private double calculateDistanceBetweenRows(Vector vector, double[][] trainData1, int currentTestRow) {
                        double sum1 = 0.0;
                        for (int i = 0; i < vector.length() - 1; i++) {
                            sum1 += Math.pow((vector.elements()[i] - trainData1[currentTestRow][i]), 2);
                        }
                        return Math.sqrt(sum1);
                    }
                }
        );

        tempCount = distances.count();
        System.out.println("\n\nDistance calculation - Done \n\n");

        //We take the distances Tuple2 RDD and find the indices of k nearest neighbours
        //it returns another Tuple2 RDD which contains the indices and the row number
        JavaPairRDD<Vector, Long> nearestNeighbourIndices = distances.mapToPair(
                new PairFunction<Tuple2<Vector, Long>, Vector, Long>() {
                    @Override
                    public Tuple2<Vector, Long> call(Tuple2<Vector, Long> distanceT2) throws Exception {
                        Vector distance = distanceT2._1();
                        double[] distanceArray = distance.elements();
                        double[] nearestNeighborIndices = new double[k];
                        for(int i=0; i<k; i++){
                            int minIndex = 0;
                            double minValue = Double.POSITIVE_INFINITY;
                            for(int j=0; j<distanceArray.length; j++){
                                if(distanceArray[j] < minValue){
                                    minValue = distanceArray[j];
                                    minIndex = j;
                                }
                            }
                            nearestNeighborIndices[i] = (double)minIndex;
                            distanceArray[minIndex] = Double.POSITIVE_INFINITY;
                        }
                        Vector vNeighbours = new Vector(nearestNeighborIndices);
                        return new Tuple2<Vector, Long>(vNeighbours, distanceT2._2());
                    }
                }
        );

        tempCount = nearestNeighbourIndices.count();
        System.out.println("\n\nNearest neighbour calculation - Done \n\n");

        //find the class labels for test data
/*        JavaRDD<Integer> predictedLabels = nearestNeighbourIndices.map(
                new Function<Tuple2<Vector, Long>, Integer>() {
                    @Override
                    public Integer call(Tuple2<Vector, Long> neighbourT2) throws Exception {
                        double[][] trainData1 = broadcastedTrainData.getValue();
                        double[] nearestNeighbourIndices = neighbourT2._1().elements();

                        int sum1 = 0;
                        for(int i=0; i<k; i++){
                            int index = (int) nearestNeighbourIndices[i];
                            sum1 += trainData1[index][dataDimension-1];
                        }
                        if(sum1*2 > k){
                            return 1;
                        }
                        else{
                            return 0;
                        }
                    }
                }
        );*/

        //calculate the predicted lables. return a tuple
        //the first element of the tuple is label, second element is row number
        JavaPairRDD<Integer, Long> plabels = nearestNeighbourIndices.mapToPair(
                new PairFunction<Tuple2<Vector, Long>, Integer, Long>() {
                    @Override
                    public Tuple2<Integer, Long> call(Tuple2<Vector, Long> neighbourT2) throws Exception {
                        double[][] trainData1 = broadcastedTrainData.getValue();
                        double[] nearestNeighbourIndices = neighbourT2._1().elements();

                        int sum1 = 0;
                        for(int i=0; i<k; i++){
                            int index = (int) nearestNeighbourIndices[i];
                            sum1 += trainData1[index][dataDimension-1];
                        }
                        if(sum1*2 > k){
                            return new Tuple2<Integer, Long>(1, neighbourT2._2());
                        }
                        else{
                            return new Tuple2<Integer, Long>(0, neighbourT2._2());
                        }
                    }
                }
        );

        Random rand = new Random();
        int randomNumber = rand.nextInt(100000);
        plabels.saveAsTextFile("output_"+randomNumber+".txt");
        System.out.println("\n\nOutput saved in the following file: " + "output_"+randomNumber+".txt in HDFS\n\n");

        tempCount = plabels.count();
        System.out.println("\n\nClass label prediction - Done \n\n");

//        printRDDTuple(testRDDZipped); // print a RDD with tuple
//        printRDDTuple(distances);

        sc.stop();
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("\n\n\n\n\nEnd of Spark KNN CLassification\nTotal time taken: " + totalTime + " ms\n\n\n");
    }

    // print a RDD with tuple
    // where tuple_1() contains a Vector
    // and tuple_2() contains a Long value
    private static void printRDDTuple(JavaPairRDD<Vector, Long> trainRDDZipped) {
        //System.out.println("\n\n\n\n\nTrainData zipped\n"+ trainData.zipWithIndex().collect().toString()+"\n\n\n\n");
        System.out.println("\n\n\n");
        for(Tuple2<Vector, Long> trainT2:trainRDDZipped.collect()){
            System.out.print(trainT2._2()+ " ");
            System.out.println(trainT2._1());
        }
    }


}
