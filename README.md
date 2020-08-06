# GibbsLDApy

这是一份GibbsLDA++的python copy版，写的内容和原版是基本一致的，改写成python的原因有两个，一个是有利于调试，更加容易看清楚程序的中间执行过程。第二是想练练手，熟悉一下原版的c++代码有利于后期工作的展开。这份代码花了两天时间写完的，所以在代码优美性上基本不用考虑，能用就行

## How to use

和c++版本的一样，接受命令行参数，你也可以在python终端中导入包后执行函数。
	
	 python LDA.py -est -alpha 0.5 -beta 0.1 -ntopics 100 -niters
     1000 -savestep 100 -twords 20 -dfile test_data/dfile

     
参数的解释在原版的英文文档中很详细了，我懒得翻译了，甚至是复制粘贴，所以请去 ./doc/GibbsLDA++Manual.pdf 里面自行查阅。


## Attention

注意事项：

  * 用来训练的文本数据一定是每个单词之间一个空格，不能多
  * test.py 文件是用来测试函数的，可以去里面看看我是怎么测试所有函数的可用性，嘲笑一下我。（里面的代码我都注释掉了）
  * inference的函数我还没有test,我在函数上面有标注not tested，在model.py里，你可以看到（用来训练新数据的，应该是能用，懒得测了）
  * 不要觉得我的矩阵声明定义很蠢，因为确实很蠢，如下...
  		
  		b = [[0]*10]*10   已经可以生成一个10 * 10 的初始化后的矩阵，而我写成了
  		       
  		self.nd = []  # int[M]
        for w in range(self.M):
            nd_row = []  # int[K]
            for k in range(self.K):
                nd_row.append(0)
            self.nd.append(nd_row)            
        这种样式，我自己也感到很羞愧，但是我不改了，懒。
	
  * 经过别人使用验证，我上面这样的矩阵定义在内存使用上有很大问题，基本可以说数据量大了就不能用了，需要使用numpy里面的矩阵重新写一遍，算了，吃一堑长一智，以后注意点了。
        
  * test_data里面的数据都是测试用例。
        
  ok,就这么多，这个算是 beta 0.0.1 ，估计也不会有新的了，大家觉得我还算辛苦的话，就 star 一个吧，我不要脸，恩。
