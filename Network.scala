package layer
class network(){
  type T = Float
  def forwards(layers:List[Layer] , n:Array[T]) : Array[T] = {
    var temp = n
    for(i<-layers){
      temp = i.forward(temp)
    }
    temp
  }
  def forwards(layers:List[Layer] , n:Array[Array[T]]) : Array[Array[T]] = {
    var temp = n
    for(i<-layers){
      temp = i.forward(temp)
    }
    temp
  }
  def backwards(layers:List[Layer] , n:Array[T]) : Array[T] = {
    var temp = n
    for(i<-layers){
      temp = i.backward(temp)
    }
    temp
  }
  def backwards(layers:List[Layer] , n:Array[Array[T]]):Array[Array[T]]={
    var temp = n
    for(i<-layers){
      temp = i.backward(temp)
    }
    temp
  }
  def updates(layers:List[Layer])={
    for(i<-layers){
      i.update()
    }
  }
  def resets(layers:List[Layer])={
    for(i<-layers){
      i.reset()
    }
  }
  def saves(layers:List[Layer],fn:String)={
    for(i<-layers){
      i.save(fn)
    }
  }
  def loads(layers:List[Layer],fn:String)={
    for(i<-layers){
      i.load(fn)
    }
  }
}
