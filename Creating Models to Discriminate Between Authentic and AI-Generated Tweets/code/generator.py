import gpt_2_simple as gpt2

model_name = "355M"
# Downloads the model
# gpt2.download_gpt2(model_name=model_name)

sess = gpt2.start_tf_sess()

gpt2.load_gpt2(sess)

"""
gpt2.finetune(sess, 
              'data/1k_teigen_tweets.csv',
              model_name=model_name,
              steps=100)

"""
gpt2.generate_to_file(sess,
              length=50,
              temperature=1.0,
              nsamples=5000,
              batch_size=20,
              prefix='<|startoftext|>',
              truncate='<|endoftext|>',
              include_prefix=False,
              destination_path='data/5k_fake_teigen_tweets.txt'
              )
