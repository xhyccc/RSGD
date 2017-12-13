package edu.mst.xionglab.rsgd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import Jama.Matrix;

public class RandomGradientObjectiveFunctionArray implements  ObjectiveFunction{
	private List<ObjectiveFunction> _functions = new ArrayList<ObjectiveFunction>();
	private static Random see=new Random(-4);

	@Override
	public double evaluation(Matrix input) {
		// TODO Auto-generated method stub
		double sum = 0;
		for(ObjectiveFunction of:this._functions)
			sum+=of.evaluation(input);
		return sum/this._functions.size();
	}


	@Override
	public Matrix gradient(Matrix input, Matrix prev) {
		// TODO Auto-generated method stub
		int index = Math.abs(Math.abs(see.nextInt())%this._functions.size());
		return this._functions.get(index).gradient(input,prev);
	}

	
}
