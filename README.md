# Objective

This Python script is like a smart to-do list manager. It uses tools from OpenAI and Pinecone to make, organize, and complete tasks. The cool thing about this system is that it makes new tasks based on what happened with the old tasks and a set goal. It uses OpenAI's language understanding skills to make new tasks based on this goal, and Pinecone to keep track of and get back task results for reference.

This README will cover the following:

- [How the program works](#how-it-works)

- [How to use the program](#how-to-use)

- [Warning about running the script continuously](#continous-script-warning)

# How It Works<a name="how-it-works"></a>

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

The task_creation_agent() function uses OpenAI's tools to generate new tasks based on the goal and the outcome of the previous task. This function takes in four inputs: the goal, the result of the previous task, the task description, and the current list of tasks. It sends a message to OpenAI's system, which then returns a list of new tasks in string format. The function then gives back these new tasks as a list of dictionaries, where each dictionary contains the name of the task.
</br>
</br>
The prioritization_agent() function uses OpenAI's tools to rearrange the task list based on priority. This function takes in one input, the ID of the current task. It sends a message to OpenAI's system, which then returns the rearranged task list as a numbered list.
</br>
</br>
The execution_agent() function is where we use OpenAI's tools. This function takes in two inputs: the goal and the task. It sends a message to OpenAI's system, which then returns the task's result. This message includes a description of the AI system's task, the goal, and the task itself. The outcome is then given back as a string.
</br>
</br>
Lastly, the script uses Pinecone to keep track of and get back task results for reference. The script sets up a Pinecone index, which is like a special database, using the table name you provide in the YOUR_TABLE_NAME variable. Pinecone is then used to save the outcomes of the task in this index, along with the task name and any extra details.

# How to Use<a name="how-to-use"></a>

To use the script, you will need to follow these steps:

# Warning<a name="continous-script-warning"></a>

This script is designed to be run continuously as part of a task management system. Running this script continuously can result in high API usage, so please use it responsibly. Additionally, the script requires the OpenAI and Pinecone APIs to be set up correctly, so make sure you have set up the APIs before running the script.
