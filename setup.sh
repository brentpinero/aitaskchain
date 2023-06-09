echo "Installing requirements.txt... "
pip install -r requirements.txt

if [ ! -f ".env" ]; then
touch .env 
fi

echo ""
echo "Enter your OpenAI API Key: "
read openai_api_key
echo "OPENAI_API_KEY=$openai_api_key" >> .env 

echo ""
echo "Please input the OpenAI model do you want to use? (gpt-3.5 turbo, gpt-4, etc.)"
read openai_model
echo "OPENAI_API_MODEL=$openai_model" >> .env 

echo ""
echo "Please enter a temperature value for the text generation. The temperature is a parameter that controls the randomness of the model's output."
echo ""
echo "If you want the output to be more deterministic and focused, choose a lower value (e.g., 0.1), and if you want the output to be more diverse and creative, choose a higher value (e.g., 1.0 or above)"
read openai_temp
echo "OPENAI_TEMPERATURE=$openai_temp" >> .env 

echo ""
echo "Enter your Serp API Key: "
read serp_api_key
echo "SERP_API_KEY=$serp_api_key" >> .env 

echo ""
echo "Enter your Pinecone API Key: "
read pinecone_api_key
echo "PINECONE_API_KEY=$pinecone_api_key" >> .env 

echo ""
echo "What region does your Pinecone environment use? (example: asia-southeast1-gcp)"
read pinecone_env
echo "PINECONE_ENVIRONMENT=$pinecone_env" >> .env 

echo ""
echo "What is the name of your pinecone index table? "
read pinecone_index_name
echo "TABLE_NAME=$pinecone_index_name" >> .env 

echo ""
echo "Now, it's time to limit the amount of iterations (or cycles) for your task chain to safeguard your wallet. More iterations means more API calls, which if left unconstrained can get expensive."
echo "Recommendation: Start out with 5, and work your way up from there."
read iterations
echo "MAX_ITERATIONS=$iterations" >> .env 

echo ""
echo "Congrats your Task Chain project is ready!"
echo "Run 'python taskchain.py' to start the program"




