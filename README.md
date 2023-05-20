# Task Chain

This Python script is like a smart to-do list manager. It uses tools from OpenAI and Pinecone to make, organize, and complete tasks. The cool thing about this system is that it makes new tasks based on what happened with the old tasks and a set goal. It uses OpenAI's language understanding skills to make new tasks based on this goal, and Pinecone to keep track of and get back task results for reference.

This README will cover the following:

- [How the program works](#how-it-works)

- [How to use the program](#how-to-use)

- [How to craft an objective](#creating-an-objective)

- [Warning about running the script continuously](#continous-script-warning)
  </br>
  </br>

# **How It Works**<a name="how-it-works"></a>

The script works by running an infinite loop that does the following steps:

1. The script first grabs the top task from our list of tasks.
   </br>
   </br>
2. It then sends this task to a part of the system called the execution agent. This agent uses OpenAI's tools to carry out the task, taking into account any relevant information.
   </br>
   </br>
3. Once the task is done, the result is enhanced with additional information and stored in Pinecone for future reference.
   </br>
   </br>
4. The system then generates new tasks and rearranges the task list based on the goal and the outcome of the previous task.
   </br>
   </br>

## **AI Chains**

 </br>
 
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

# How to Use<a name="how-to-use"></a>

Before you start, ensure that you have Python (3.6 or higher), Git, and pip installed on your machine.

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

# Crafting an Objective<a name="creating-an-objective"></a>

To effectively instruct the AI Task Chain, your objectives need to be clear, concise, and purposeful. Here's a simplified guide on how to craft a beneficial objective:
</br>
</br>

## Use a Clear Prefix

When defining your objective, prefix it with "My objective is". This statement ensures clarity and helps direct the AI in a more focused manner.
</br>
</br>

## Identify Your Goal

Clearly outline your end goal. Whether it's generating content or fetching data, ensure the desired outcome is stated in simple language.
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

This script is designed to be run continuously as part of a task management system. Running this script continuously can result in high API usage, so please use it responsibly. Additionally, the script requires the OpenAI and Pinecone APIs to be set up correctly, so make sure you have set up the APIs before running the script.
