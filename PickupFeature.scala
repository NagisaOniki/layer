package layer
class PickupFeature(
  val imageH:Int,
  val imageW:Int,
  val rgb:Int
)extends Layer{

  def forward(image:Array[Float])={
    val ln = 100
    //----network--------------
    val N = new network()
    val af_ep = 0.01f
    val af_p1 = 0.9f
    val af_batch = 0
    val con_ep = 0.01f
    val con_p1 = 0.9f
    val OH = 15
    val OW = 15
    val OC = 10

    val before = List(
      new ConvolutionParam(3,imageH,imageW,rgb,10,con_ep,con_p1),new ReLU(),
      new Pooling(2,10,30,30),
      new ConvolutionParam(2,15,15,10,20,con_ep,con_p1),new ReLU(),
      new Pooling(2,20,14,14),
      new AffineParam(7*7*20,14*14*20,af_ep,af_p1,af_batch),
      new ReLU(),
      new AffineParam(14*14*20,imageH*imageW*rgb,af_ep,af_p1,af_batch)
    )
    val after = List(
      new ConvolutionParam(3,imageH,imageW,rgb,10,con_ep,con_p1),new ReLU(),
      new Pooling(2,10,30,30),
      new ConvolutionParam(2,15,15,10,20,con_ep,con_p1),new ReLU(),
      new Pooling(2,20,14,14),
      new AffineParam(7*7*20,14*14*20,af_ep,af_p1,af_batch),
      new ReLU(),
      new AffineParam(14*14*20,imageH*imageW*rgb,af_ep,af_p1,af_batch)
    )

    var newImage = new Array[Float](imageH*imageW*rgb)
    //----learning------------
    for(c<-0 until ln){
      //----forward--------------
      val A = N.forwards(before , image)
      val y = N.forwards(after , A)
      //----miss---------------
      var d = new Array[Float](imageH*imageW*rgb)
      var t = new Array[Float](imageH*imageW*rgb)
      for(i<-0 until imageH*imageW*rgb){
        d(i) = y(i) - image(i)
      }
      //----backward------------
      val Z = N.backwards(after.reverse , d)
      N.backwards(before.reverse , Z)
      //----update------------
      N.updates(before)
      N.updates(after)
      //----GlobalAveragePooling----
      val GAP = new GlobalAveragePooling(imageH,imageW,rgb)
      val alpha = GAP.forward(Z)
      //----G----------------
      var G = new Array[Float](imageH*imageW)
      for(i<-0 until rgb){
        for(p<-0 until imageH ; q<-0 until imageW){
          G(p*imageW+q) += alpha(i) * A(i*imageH*imageW + p*imageW+q)
        }
      }
      val relu = new ReLU()
      G = relu.forward(G)
      for(c<-0 until rgb){
        for(i<-0 until imageH ; j<-0 until imageW){
          newImage(c+i*imageW+j) = image(c+i*imageW+j) * G(i*imageW+j)
        }
      }
      //----reset-------------
      N.resets(before)
      N.resets(after)
    }//ln
    newImage
  }//def

  def forward(image:Array[Array[Float]])={
    image.map(forward)
  }

  def backward(image:Array[Float])={
    image
  }
  def backward(image:Array[Array[Float]])={
    image.reverse.map(backward).reverse
  }
  def update(){}
  def reset(){}

}//class
