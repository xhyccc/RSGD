package edu.mst.xionglab.rsgd;

import Jama.Matrix;

public interface SingleVariableObjectiveFunction {
	public double evaluation(Matrix input);
	public Matrix gradient(Matrix input, Matrix prev);
}
