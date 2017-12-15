package edu.mst.xionglab.rsgd;

import Jama.Matrix;

public class MultiVariableGradientDescent {
	public static double convergence = 1.0e-4;
	public static int maxIterations = 10000;
	private MultiVariableObjectiveFunction _of;
	private Matrix[] _parameter = null;
	private Matrix[] _parameterPrev = null;
	private double _learningRate = 0.01;
	private int count = 0;

	public MultiVariableGradientDescent(MultiVariableObjectiveFunction of) {
		this._of = of;
	}

	public Matrix[] getPreviousParameter() {
		return this._parameterPrev;
	}

	public MultiVariableGradientDescent(MultiVariableObjectiveFunction of, Matrix[] parameter) {
		this(of);
		this._parameter = parameter;
	}

	public MultiVariableGradientDescent(MultiVariableObjectiveFunction of, Matrix[] parameter, double learningRate) {
		this(of, parameter);
		this._learningRate = learningRate;
	}

	public double largest(double[] datums) {
		double max = datums[0];
		for (int i = 1; i < datums.length; i++) {
			if (datums[i] > max) {
				max = datums[i];
			}
		}
		return max;
	}

	public double iterate() {
		count++;
		Matrix[] gradient = this._of.gradient(this._parameter, this._parameterPrev);
		this._parameterPrev = this._parameter;
		double[] errors = new double[this._parameter.length];
		for (int i = 0; i < this._parameter.length; i++) {
			this._parameter[i] = this._parameter[i].plus(gradient[i].times(-1 * this._learningRate));
			errors[i] = gradient[i].norm2();
		}
		System.out.println(count + "\t" + this._of.evaluation(this._parameter) + "\t" + largest(errors));
		return largest(errors);
	}

	public double iterate(Matrix[] parameter) {
		this._parameter = parameter;
		return this.iterate();
	}

	public double iterate(Matrix[] parameter, double learningRate) {
		this._learningRate = learningRate;
		return this.iterate(parameter);
	}

	public Matrix[] getResult() {
		return this._parameter;
	}

	public boolean continueIterate() {
		if (maxIterations < count)
			return false;
		double error = iterate();
		if (error > MultiVariableGradientDescent.convergence) {
			return true;
		} else
			return false;
	}

	public boolean continueIterate(Matrix[] parameter) {
		this._parameter = parameter;
		return continueIterate();
	}

	public boolean continueIterate(Matrix[] parameter, double learningRate) {
		this._learningRate = learningRate;
		return continueIterate(parameter);
	}

	public Matrix[] GDProcedure() {
		while (this.continueIterate());
		return this.getResult();
	}

	public Matrix[] GDProcedure(Matrix[] init) {
		this._parameter = init;
		return GDProcedure();
	}

	public Matrix[] GDProcedure(Matrix[] init, double[] rates) {
		int loop = 0;
		while (this.continueIterate(this.getResult(), rates[loop++]));
		return this.getResult();
	}
}
