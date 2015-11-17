package linearML;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class LinearRegressionClassifier implements Serializable {

	private static final long serialVersionUID = 7197882259463196104L;

	private int dimension;
	private String[] classes;
	private Matrix theta; // The Matrix we want to build which contains the unknowns
	
	
	private int repetitions =3;
	private double learnRate = 1;
	private double initial_value =0.5;

	public LinearRegressionClassifier(int dimension, String[] classes) {
		this.dimension = dimension;
		this.classes = classes;
		theta = new Matrix(dimension, classes.length); // Create an empty matrix
		fillMatrix(initial_value);
	}
	
	public LinearRegressionClassifier(int dimension, String[] classes, double learning_rate, int numRepetitions) {
		this(dimension, classes);
		repetitions = numRepetitions;
		learnRate = learning_rate;
	}
	
	//fills matrix with the same value (initial value before starting to do gradient descent)
	private void fillMatrix(double i)
	{
		for(int r =0; r<dimension; r++)
		{
			for(int c =0; c<classes.length;c++)
			{
				theta.set(r, c, i);
			}
		}
		
	}

	public String classify(double[] data) {
		Map<String, Double> distribution = classDistribution(data);

		double maxValue = -1;
		String max = "";

		for (String key : distribution.keySet()) {
			double temp = distribution.get(key);
			if (temp > maxValue) {
				maxValue = temp;
				max = key;
			}
		}

		return max;
	}

	public void train(double[][] data, String[] labels) throws FileNotFoundException, UnsupportedEncodingException {
		
		int trainSize = data.length;
		double gradient =0;
		double tempTheta=0;
		//For every class
		for(int label =0; label<classes.length; label++)
		{
			//We repeat the stochastic gradient descent multiple times
			for(int i =0; i<repetitions; i++)
			{
				for(int t =0; t<trainSize;t++){
					//For every unknown/theta 
					for(int r=0; r<dimension; r++)
					{						
						//For every theta we want to do the following thetaj = thethaj - (theta1*x1+theta2*x2+...+thetan*xn -label)xj
						//We use the xs and label of the training data we're currently using 
						gradient = 0;
						tempTheta = theta.get(r, label);
						for(int r2=0; r2<dimension;r2++)
						{
							//thetai*xi added up
							gradient += data[t][r2]*theta.get(r2, label);
						}
						// - yi
						if(labels[t].equals(classes[label]))
							gradient -=1.0d;
                                                if(t%1000 == 0)
                                                    System.out.println("sum and multiplication for  "+(r+1)+" is "+gradient);
						//*xj
						gradient *= data[t][r];
						//*-learnRate
						gradient *=learnRate;
                                                if(t%1000 == 0)
                                                    System.out.println("gradient for  "+(r+1)+" is "+gradient);
						gradient /= data.length;
						tempTheta -= gradient;
						theta.set(r, label, tempTheta);
                                                if(t%1000 == 0)
                                                    System.out.println("Theta "+(r+1)+" is "+tempTheta);
					}
				}
			}
		}
	}

	public Map<String, Double> classDistribution(double[] data) {
		Map<String, Double> distribution = new HashMap<String, Double>();

		Matrix dataM = new Matrix(new double[][] { data });

		// Multiply the 1xn data matrix by the nxm theta matrix to result in our
		// result of 1x10 probabilities matrix
		Matrix distributionM = dataM.times(theta);

		// Transform back to matrix
		double[] result = distributionM.getArray()[0];

		// fill the hashmap with the keys and probabilities
		for (int i = 0; i < classes.length; i++) {
			distribution.put(classes[i], result[i]);
		}

		return distribution;
	}

	public int getDimension() {
		return dimension;
	}

	public String[] getClasses() {
		return classes;
	}

	public Matrix getTheta() {
		return theta;
	}

}
