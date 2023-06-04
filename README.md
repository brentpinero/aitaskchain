# Task Chain Manual

AI Task Chain is a Python tool that uses APIs such as OpenAI, Pinecone, and SerpAPI to intelligently execute a series of tasks to achieve a specific given objective. It uses OpenAI's language comprehension capabilities to generate new tasks based on the results of previous tasks. Pinecone is used to store and retrieve task results, which helps AI Task Chain make better decisions. The system loops continuously until the objective is achieved or the maximum number of iterations is reached.

This README will cover the following:

- [Setup Instructions](#setup_instructions)

- [How the program works](#how-it-works)

- [How to craft an objective](#creating-an-objective)

- [Warning about running the script continuously](#continous-script-warning)
  </br>
  </br>

# Setup Instructions<a name="setup_instructions"></a>

Before you start, ensure that you have Python (3.6 or higher), Git, and pip installed on your machine. If you're on Windows or Linux, make sure that you have the latest version of Ubuntu installed on your machine.

1. Clone the repository using git: </br>
   `git clone https://github.com/brentpinero/aitaskchain.git`

2. Navigate to the project's directory: </br>
   `cd aitaskchain`

3. Now, execute the setup script to install the required dependencies and to create the environment file (.env) that will store your API keys and other required parameters: </br>
   `./setup.sh`
   </br>
   </br>
   The setup.sh script will ask you for various inputs:

   -OpenAI API Key</br>
   -OpenAI Model (e.g., gpt-3.5 turbo, gpt-4)</br>
   -Temperature for text generation</br>
   -Serp API Key</br>
   -Pinecone API Key</br>
   -Pinecone Environment</br>
   -Pinecone Index Table</br>
   -Maximum number of iterations</br>
   </br>
   Please make sure you have these details at hand, as they are necessary for the proper functioning of the Task Chain.
   </br>
   </br>

4. Once you have successfully run `setup.sh`, your environment will be ready, and you can execute the AI Task Chain using:</br>
   </br>
   `python taskchain.py`

</br>

# **How It Works**<a name="how-it-works"></a>

The taskchain.py script uses APIs such as OpenAI, Pinecone, and SerpAPI to automate task completions. Here is a brief description of the steps:

1. The script creates a list of tasks based on the provided objective, and then identifies and fetches the top priority task for accomplishing the objective.
   </br>
   </br>
2. The script then sends this task to a part of the system called the execution agent. In order to carry out the task, the execution agent uses a SerpAPI tool to scrape relevent information from Google Search.
   </br>
   </br>
3. After execution, the result is stored using Pinecone, a vector database for vector search.
   </br>
   </br>
4. New tasks are created based on the result and added to the task queue. The task queue is reprioritized.
   </br>
   </br>
5. The script repeats these steps until the task list is empty or the maximum number of iterations has been reached.
   </br>
   </br>

## **AI Chains**

### **Task Creation Chain**

The task_creation_agent() function uses OpenAI's tools to generate new tasks based on the goal and the outcome of the previous task. This function takes in four inputs: the goal, the result of the previous task, the task description, and the current list of tasks. It sends a message to OpenAI's system, which then returns a list of new tasks in string format. The function then gives back these new tasks as a list of dictionaries, where each dictionary contains the name of the task.
</br>
</br>

### **Prioritization Chain**

The prioritization_agent() function uses OpenAI's tools to rearrange the task list based on priority. This function takes in one input, the ID of the current task. It sends a message to OpenAI's system, which then returns the rearranged task list as a numbered list.
</br>
</br>

### **Execution Chain**

The execution_agent() function is where we use OpenAI's tools. This function takes in two inputs: the goal and the task. It sends a message to OpenAI's system, which then returns the task's result. This message includes a description of the AI system's task, the goal, and the task itself. The outcome is then given back as a string.
</br>
</br>

## **Keep Track of Tasks**

Lastly, the script uses Pinecone to keep track of and get back task results for reference. The script sets up a Pinecone index, which is like a special database, using the table name you provide in the YOUR_TABLE_NAME variable. Pinecone is then used to save the outcomes of the task in this index, along with the task name and any extra details.
</br>
</br>

# Crafting an Objective<a name="creating-an-objective"></a>

To effectively instruct the AI Task Chain, your objectives need to be clear, concise, and purposeful. Here's a simplified guide on how to craft a beneficial objective:
</br>
</br>

## Use a Clear Prefix

When defining your objective, prefix it with "My objective is". This statement ensures clarity and helps direct the AI in a more focused manner.
</br>
</br>

## Identify Your Goal

Clearly outline your end goal. Whether it's generating content or fetching data, ensure the desired outcome is stated in specific and concise language.
</br>
</br>

## Incorporate Constraints

If any specific requirements or constraints are needed, include them. For example, using or excluding certain keywords.
</br>
</br>

## Test and Iterate

Use your objective with the AI Task Chain, and refine based on the results. Remember, crafting objectives is an iterative process - you'll refine your technique over time.
</br>
</br>
Remember, the better your objective is defined, the better the AI can fulfill your request.

</br>

# Warning<a name="continous-script-warning"></a>

This script is designed to be run continuously as part of a task management system. Running this script continuously can result in high API usage, so please use it responsibly.
