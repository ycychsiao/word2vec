//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <thread>
#include <iostream>
#include <time.h>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

#define MAX_STRING 100//string ?型的最大?度
#define EXP_TABLE_SIZE 1000//?里是用?求sigmoid函?,使用的是一种近似的求法，
#define MAX_EXP 6//只要求球???６的即可
#define MAX_SENTENCE_LENGTH 1000//句子最大?度,及包含??
#define MAX_CODE_LENGTH 40//huffman?程中?word?行按??的huffman code,每??的最大?度?４０，也可理解??的高度不?超?２０
//#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;//??
  int *point;//huffman???????的路?
  char *word, *code, codelen, *decomp, *token;;//次??，huffman??，???度
};

char train_file[MAX_STRING], output_file[MAX_STRING];//??文件和?出文件
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];//保存??表和?取??表的文件。格式：?　??
struct vocab_word *vocab;//?????
//binary?行二?制文件?取?入，cbow???BAG OF WORDS?构，
//window?窗口大小 
//min_count???下限，小于?下限忽略；
//num_threads??程?（多?程?每??程??部分??文件，即?整???文件均分?多??程，
//多??程更新所有的??（更新???的?取?突可以忽略），其他??等所有?程共享），
//min_reduce????行??
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 7, min_reduce = 1;
int *vocab_hash;//?的hash表
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
// train_words ??的????（??累加）
// word_count_actual 已???完的word??
// file_size ??文件大小，ftell得到
// classes ?出word clusters的???
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;

//syn0?所有word的vector，syn1?huffman tree中所有?部??的vector，
//syn1neg? negative sampling???sampling的vector表示，
//expTable?exp（）函?的离散表示，?了?省??。
real *syn0, *syn1neg, *expTable;

clock_t start;
//hs表示hierarchical softmax，即?次化softmax替代原?的softmax??少?算??，加速??，默?不采用；
//negative?negative sampling，默?采用
int hs = 0, negative = 5;
const int table_size = 1e8;

//negative sampling? 的分布table
int *table;
unordered_map<string,vector<char*>> partition;//NEW
int sim = 10;

void sample_of_similar(int *arr){
	int random, tmp, i,s;
	srand(time(NULL));
	s = vocab_size/4;
	for(i = 0 ; i < s; i++)	arr[i]=i;
	for(i = 0; i < s;i++){
		random = rand() % s;
		tmp = arr[random];
		arr[random] = arr[i];
		arr[i]=tmp;
	}
	return;	
}

//int word_part(int *arr, int len, int c){
//	int local;
//	for(int i=0; i<len; i++){
//		if( c-arr[i] < 0 )  
//	} 
//	return ;
//}

void InitUnigramTable() {
  printf("enter InitUnigramTable()\n");
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
//?取一???，假?每???以空格或者tab或者?行符??尾
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");//strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
//??的hash值
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
//返回word 在??hash表中的的位置
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
//?文件中?取一???并返回它在??hash表中的下?
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
//向??表中添加一???
int AddWordToVocab(char *word, unordered_map<string, char*> &decomposition, FILE *fout) {	
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  if(vocab[vocab_size].word==NULL) printf("filed to calloc word\n");
  
  vocab[vocab_size].token = (char *)calloc(3*length, sizeof(char));//NEW
  if(vocab[vocab_size].token==NULL) printf("failed to calloc token\n");
  
  vocab[vocab_size].decomp = (char *)calloc( 10*length , sizeof(char));//NEW
  if(vocab[vocab_size].decomp==NULL) printf("failed to calloc decomp\n");
   

  string tok="";
  int cur = 0,unit = 0;
  while( cur < length-1 ){  
    char t = word[cur];
    string lines="";
    
   	if((t&0xE0) == 0xE0){  //3byte// 11 10 00 00
 	    lines = lines + t + word[cur+1] + word[cur+2];
       	cur += 3;
    }else if((t&0xC0) == 0xC0){//2byte//11 00 00 00
        lines = lines + t + word[cur+1];
        cur += 2;
    }else if(0x00 <= t && t <= 0x7f){//1byte
    	lines = lines + t;
    	cur++;
    }

    if(decomposition[lines]!=NULL){
		char *tpp=decomposition[lines];
		strcat(vocab[vocab_size].decomp, tpp);// lines = per token
		unit++;
	}
    lines = lines + ' ';
    tok = tok + lines ;
   
  }
  
  if(unit==0) vocab[vocab_size].decomp=NULL;
  
  strcpy(vocab[vocab_size].token, tok.c_str());
  strcpy(vocab[vocab_size].word, word);
  
//  printf("%s %s %s\n",vocab[vocab_size].word,vocab[vocab_size].decomp,vocab[vocab_size].token);
//  fprintf(fout,"%s %s %s\n",vocab[vocab_size].word,vocab[vocab_size].decomp,vocab[vocab_size].token);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  
  // Reallocate memory if needed 
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}


int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
//???表?行排序，利用??出?的?率
//并去掉低??
void SortVocab() {
  printf("enter SortVocab\n");
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  //</s> 是一?特殊的字符
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  //?重排之后的??hash表?行更新
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  //去掉低??
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
      free(vocab[a].decomp);
      free(vocab[a].token);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  printf("freeed\n");
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
//
void ReduceVocab() {
  printf("ReduceVocab\n");
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) 
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      vocab[b].decomp = vocab[a].decomp;
      vocab[b].token = vocab[a].token;
      b++;
    } else{
    	free(vocab[a].word);
    	free(vocab[a].decomp);
    	free(vocab[a].token);
	} 
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  //清空?出???
  fflush(stdout);
  min_reduce++;
}


//???文件中得到???率表
void LearnVocabFromTrainFile(unordered_map<string, char*> &decomposition,FILE *fout) {
  printf("enter\n");
  char word[MAX_STRING];
  FILE *fin;
  long long a, i,j;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  //第一??特殊??
  vocab_size = 0;
  AddWordToVocab((char *)"</s>",decomposition,fout);
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word,decomposition,fout);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7){ 
      printf("ReduceVocab()\n");
	  ReduceVocab();
    }
  }
  printf("SortVocab()\n");
  SortVocab();
  string ll;
  for (a = 0; a < vocab_size; a++){
    if (vocab[a].decomp!=NULL) {
        j=0;	
        while( j < strlen(vocab[a].decomp) ){
      		char t=vocab[a].decomp[j];
      		if(t != ' ' ) 	ll += t;
      		else{
      			partition[ll].push_back(vocab[a].word);
      			ll="";
      		}
      		j++;
        }
    } 
  }
  printf("run throuth learnfromvocabularyfile\n");
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab(unordered_map<string,char*> &decomposition,FILE *fout) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word,decomposition,fout);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

//
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  //posix_memalign是用???函?
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  //Hierarchical Softmax 模型

  //Negative Sampling 模型
  if (negative > 0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++)
        for (b = 0; b < layer1_size; b++)
        	syn1neg[a * layer1_size + b] = 0;
		     
  }
  //?机初始化，?怎么看懂
  for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      }

}

void TrainModelThread(int id,FILE *fout) {
	long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  	long long l1, l2, l3, l4, c, target, label, local_iter = iter;
  	unsigned long long next_random = (long long)id;
    real f, g, f2, g2, aver, grad, max_grad = 0, min_grad = 0;
  	clock_t now;
  	int j,len,times,num,z,x,random[vocab_size/4]={0};
	string ll;
	
	//for(int r=0 ; r<5 ; r++ ) random[r]=-10;
	
 	real *neu1 = (real *)calloc(layer1_size, sizeof(real)); 
  	real *neu1e = (real *)calloc(layer1_size, sizeof(real)); 
  	real *tmp =(real *)calloc(layer1_size,sizeof(real));
  	real *tmp1 = (real *)calloc(layer1_size,sizeof(real));

  	FILE *fi = fopen(train_file, "rb");

  	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
		
	sample_of_similar(random);

  	while (1) {
    	if (word_count - last_word_count > 10000) {
      		word_count_actual += word_count - last_word_count;
      		last_word_count = word_count;
      		if ((debug_mode > 1)) {
        		now=clock();
        		printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         		word_count_actual / (real)(iter * train_words + 1) * 100,
         		word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        		fflush(stdout);
      		}
      		alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1)); 
      		if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    	}
    	//
    	if (sentence_length == 0) {
    		
      		while (1) {
        		word = ReadWordIndex(fi);
        		if (feof(fi)) break;  
        		if (word == -1) continue; 
        		word_count++;
        		if (word == 0) break; //是特殊?</s>
        		
        		if (sample > 0) {
          			real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          			next_random = next_random * (unsigned long long)25214903917 + 11;
          			if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        		}
        		sen[sentence_length] = word;
        		
        		sentence_length++;
        		if (sentence_length >= MAX_SENTENCE_LENGTH)   break;
      		}
      		sentence_position = 0;  // ?前??在?前句中的index，起始值?0
    	}
    
		if (feof(fi) || (word_count > train_words / num_threads)) {
      		word_count_actual += word_count - last_word_count;
      		local_iter--;
			sample_of_similar(random);
			if( local_iter <= iter ){
				if( id == 0){
					//for ( a = 0;a<vocab_size;a++){
					fprintf(fout,"%lld %lld %d\n",vocab_size,layer1_size,iter-local_iter+1);
					for( a = 0; a < vocab_size ; a++){
						fprintf(fout,"%s ",vocab[a].word);
						for( b = 0 ; b < layer1_size ; b++){
							fprintf(fout,"%lf ",syn0[a*layer1_size+b]);
						}
						fprintf(fout,"\n");
					}
				}
			}
      		if (local_iter == 0) break;
      		word_count = 0;
      		last_word_count = 0;
     		sentence_length = 0;
			//printf("%d\n",local_iter);
			//printf("min_grad:%f max_grad:%f",min_grad,max_grad);
      		fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      		continue;
    	}
		//if(local_iter != iter)	printf("%d\n",local_iter);
		word = sen[sentence_position];
    	if (word == -1) continue;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    	for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    	for (c = 0; c < layer1_size; c++) tmp[c] = 0;

    	next_random = next_random * (unsigned long long)25214903917 + 11;
    	b = next_random % window; //?机取窗口 
		
    	l3 = sen[sentence_position] * layer1_size;
    	num = 0;
		j = 0;
    	//printf("%s ",vocab[word].word);
		while( j < strlen(vocab[word].decomp) ){
    		char t = vocab[word].decomp[j];
      		if(t == ' ') num++;
      		j++;
		}
	
		if( num != 0 and local_iter <= iter -1 ){
    		string sampled_word[num];
    		int lens[num];
    		x = 0;
        	j = 0;
    		while( j < strlen(vocab[word].decomp) ){
    			char t = vocab[word].decomp[j];
      			if(t != ' ')	ll += t;
				else{
					sampled_word[x]=ll;
					lens[x]=static_cast<int>(partition[ll].size());
//					printf("lens[x]:%d\n",lens[x]);
					ll="";
					x++;
				}
      			j++;
			}
//			printf("%s %s ",vocab[word].word,vocab[word].decomp);
	    	//printf("x:%d num:%d\n",x,num);	
			aver = 0;
			int res[sim], s = 0, cnt = 0, lens_acc[num+1], t, o, u;
			real ni = 0, nj = 0, allni[sim], allnj[sim], norm, sum = 0, allsum[sim]; 
			long long alll4[sim];
			
			lens_acc[0] = 0;
			for ( z = 1 ; z < num+1 ; z++){
				lens_acc[z] = lens_acc[z-1] + lens[z-1]; //個數 
			}
			
			if( lens_acc[num] != 0){
				if(lens_acc[num] >= 100) 	o = 100;
				else 	o = lens_acc[num];
				u = 0;
				while( s < sim){
					x = 0;
					if( random[cnt] < lens_acc[num]){
						while((random[cnt] - lens_acc[x]) >= 0 and x < num + 1 ){
//							printf("cnt:%d x:%d lens_acc:%d res:%d random:%d\n",cnt,x,lens_acc[x],random[cnt]-lens_acc[x],random[cnt]);
							x++;
						}

						if( x != 0 ) 	x = x-1;
						t = random[cnt] - lens_acc[x];

						l4 = SearchVocab( partition[sampled_word[x]].at(t))*layer1_size;

						for (c = 0; c < layer1_size; c++){ 
							ni += syn0[ c + l3 ]*syn0[ c + l3 ] ;
							nj += pow(syn0[ c + l4], 2 );
    						sum = sum + syn0[ c+l3 ]* syn0[ c+l4 ]; //inner procuct
						}
						ni = sqrt(ni);
						nj = sqrt(nj);
						//sum = sum / (ni*nj);
						if (sum >= 0.5 and l4 != l3){//0.5?
							allsum[s]=sum;
							allni[s]=ni;
							allnj[s]=nj;
							alll4[s]=l4;
							s++;
							//printf("s:%d sum:%f ",s,sum);
							//fprintf(fout,"word:%s similar:%s cossim:%f\n",vocab[word].word,vocab[(l4/200)].word,sum);
						}
						sum = 0;
						ni = 0;
						nj = 0;
						cnt ++;
						u++;
					}
					else cnt++;
					if( u  == o-1 )	break;
				}

				for(int y = 0 ; y < s; y++) {
					aver++;
					ni = allni[y];
					
					nj = allnj[y];
					
					sum = allsum[y];
					
					l4 = alll4[y];
					
					//dis = alldis[y];
					for (c = 0; c < layer1_size; c++){ //NEW
						o = syn0[c+l3]*syn0[c+l4];
    					tmp1[c] = 2 * sum * (- (o* syn0[c+l3])/(ni * ni) + syn0[c+l4]/(ni * nj));//+ o*sum/dis
						norm += tmp1[c] * tmp1[c];
					}
					norm = sqrt(norm);
					//for( c = 0; c < layer1_size; c++){
					//	if( norm < 1 ) tmp[c] += tmp1[c]*0.001*alpha; 
					//	else tmp[c] += tmp1[c]*0.001*alpha*1/norm;
					//}
					//int rr=rand()%100;
					//if(rr > 90)	printf("%lf\n",norm);
					if(norm > max_grad)		 max_grad = norm;
					norm = 0;
				}
  			}
		}
		if( aver!=0 ){
			for (c = 0; c < layer1_size; c++){
				tmp[c] = tmp[c] / aver;	
			} 
		}
    	if (cbow) {  
      	 //
		}//cbow end
		//`
		else {  //train skip-gram
		
			for (a = b; a < window * 2 + 1 - b; a++) {
				if (a != window) {
        			c = sentence_position - window + a;
        			if (c < 0) continue;
        			if (c >= sentence_length) continue;
        			last_word = sen[c];
        			if (last_word == -1) continue;
        			l1 = last_word * layer1_size;
					for ( c=0; c<layer1_size; c++) neu1e[c]=tmp[c];
  					//
  					if (negative > 0) for (d = 0; d < negative + 1; d++) {
          				if (d == 0) {
            				target = word;
            				label = 1;
          				} else {
            				next_random = next_random * (unsigned long long)25214903917 + 11;
            				target = table[(next_random >> 16) % table_size];
            				if (target == 0) target = next_random % (vocab_size - 1) + 1;
            				if (target == word) continue;
            				label = 0;
          				}
          				l2 = target * layer1_size;
          				f = 0;
          				for (c = 0; c < layer1_size; c++) {
          					f += syn0[c + l1] * syn1neg[c + l2];
						}
          				if (f > MAX_EXP) g = (label - 1) * alpha;
          				else if (f < -MAX_EXP) g = (label - 0) * alpha;
          				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          				
          				for (c = 0; c < layer1_size; c++){
          					neu1e[c] += g * syn1neg[c + l2];
						}
          				for (c = 0; c < layer1_size; c++){
          					syn1neg[c + l2] += g * syn0[c + l1];
						}
        			}
        			//NEW
        			// Learn weights input -> hidden
        			for (c = 0; c < layer1_size; c++){
        				syn0[c + l1] += neu1e[c];
					} 
				}
			}
			
		}
		
		sentence_position++;
    	if (sentence_position >= sentence_length) {
      		sentence_length = 0;
      		continue;
    	}
	}
	printf("%lf\n",max_grad);
  	fclose(fi);
  	free(neu1);
  	free(neu1e);
  	pthread_exit(NULL);
}

//
void TrainModel(unordered_map<string, char*> &decomposition,FILE *fout) {
  long a, b, c, d;
  FILE *fo;
  //pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  //?始的??速率
  starting_alpha = alpha;
  
  //?向量初始化
  
  if (read_vocab_file[0] != 0) ReadVocab(decomposition,fout); else LearnVocabFromTrainFile(decomposition,fout);
  if (save_vocab_file[0] != 0) SaveVocab();
  
  if (output_file[0] == 0) return;
  //
  InitNet();
  //?于?采?，?全采?初始化表
  if (negative > 0) InitUnigramTable();
  //
  start = clock();
  thread t[num_threads];
  printf("thread\n");
  for (a = 0; a < num_threads; a++) t[a]=thread(TrainModelThread,a,fout);
  for (a = 0; a < num_threads; a++) t[a].join();
  printf("endthread\n");
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++)
        fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++)
        fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

//?找??，通??比str和argv[a]，?有返回-1
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

/*主程序*/
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  //EXP_TSBLE_SIZE = 1000，MAX_EXP = 6
  //?[-６，６]均分成1000等份,
  
    
  
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  
    char word2[MAX_STRING];
	FILE *finn = fopen("decomposition.txt","rb");
	FILE *fipp = fopen("partition.txt","rb");
	FILE *fout = fopen("file3.txt","wb");
	unordered_map<string ,char*> decomposition;//後在考慮要不要改成hash 
//	unordered_map<string,vector<char*>> partition;
	string line,first;	
	int l=0,f=0;
    while(!feof(finn)){//decomposition
	   	ReadWord(word2,finn);
		if(l>=f){
			first.assign(word2,strlen(word2));
			f++;
			continue;
		}
		if(strcmp(word2,"</s>")){
			line = line + word2 + ' ' ;
		}
		else{
			
			char *tmp=new char[line.length() + 1];
			strcpy(tmp, line.c_str());
			decomposition[first]=tmp;
			l++;
			line="";
		} 
  	}	 
		
			
	while(!feof(fipp)){
		ReadWord(word2,fipp);
		if(strcmp(word2,"</s>")){
			string tmp=word2;
			partition[word2] = vector<char*>();
		}	
	}
  //`printf("end"); 
  TrainModel(decomposition,fout);
  return 0;
}

