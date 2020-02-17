package layer
import breeze.linalg._
import pll.utils.RichArray._
abstract class Layer {
  type T = Float
  def forward(x:Array[T]) : Array[T]
  def backward(x:Array[T]) : Array[T]
  def forward(x:Array[Array[T]]) : Array[Array[T]]
  def backward(x:Array[Array[T]]) : Array[Array[T]]
  //def forward(x:DenseVector[T]) : DenseVector[T]
  //def backward(x:DenseVector[T]) : DenseVector[T]
  def update() : Unit
  def reset() : Unit
  def load(fn:String) {}
  def save(fn:String) {}
  def add_b(Z:Array[T] , fn:String) : Array[T] = {
    new Array[T](Z.size)
  }
  def set_ep(eps:Float){}
}
