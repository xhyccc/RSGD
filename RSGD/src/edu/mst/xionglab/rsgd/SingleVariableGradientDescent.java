package edu.mst.xionglab.rsgd;

import Jama.Matrix;

public class SingleVariableGradientDescent {
	public static double convergence = 1.0e-4;
	public static int maxIterations = 10000;
	private SingleVariableObjectiveFunction _of;
	private Matrix _parameter = null;
	private Matrix _parameterPrev = null;
	private double _learningRate = 0.01;
	private int count = 0;

	public SingleVariableGradientDescent(SingleVariableObjectiveFunction of) {
		this._of = of;
	}
	
	public Matrix getPreviousParameter(){
		return this._parameterPrev;
	}

	public SingleVariableGradientDescent(SingleVariableObjectiveFunction of, Matrix parameter) {
		this(of);
		this._parameter = parameter;
	}

	public SingleVariableGradientDescent(SingleVariableObjectiveFunction of, Matrix parameter, double learningRate) {
		this(of, parameter);
		this._learningRate = learningRate;
	}

	public double iterate() {
		count++;
		Matrix gradient = this._of.gradient(this._parameter,this._parameterPrev);
		this._parameterPrev = this._parameter;
		this._parameter = this._parameter.plus(gradient.times(-1 * this._learningRate));
		
		System.out.println(count+"\t"+this._of.evaluation(this._parameter)+"\t"+gradient.norm2());
		return gradient.norm2();
	}

	public double iterate(Matrix parameter) {
		this._parameter = parameter;
		return this.iterate();
	}

	public double iterate(Matrix parameter, double learningRate) {
		this._learningRate = learningRate;
		return this.iterate(parameter);
	}

	public Matrix getResult() {
		return this._parameter;
	}

	public boolean continueIterate() {
		if (maxIterations < count)
			return false;
		double error = iterate();
		if (error > SingleVariableGradientDescent.convergence) {
			return true;
		} else
			return false;
	}

	public boolean continueIterate(Matrix parameter) {
		this._parameter = parameter;
		return continueIterate();
	}

	public boolean continueIterate(Matrix parameter, double learningRate) {
		this._learningRate = learningRate;
		return continueIterate(parameter);
	}

	public Matrix GDProcedure() {
		while (this.continueIterate())
			;
		return this.getResult();
	}

	public Matrix GDProcedure(Matrix init) {
		this._parameter = init;
		return GDProcedure();
	}

	public Matrix GDProcedure(Matrix init, double[] rates) {
		int loop = 0;
		while (this.continueIterate(this.getResult(), rates[loop++]))
			;
		return this.getResult();
	}
}
