pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                sh 'echo "Tests would go here"'
                // Add actual tests when you have them
                // sh 'python -m pytest'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t data-analysis-chatbot:latest .'
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'echo "Deployment steps would go here"'
                // Example deployment commands:
                // sh 'docker stop data-analysis-chatbot || true'
                // sh 'docker rm data-analysis-chatbot || true'
                // sh 'docker run -d -p 8501:8501 --name data-analysis-chatbot data-analysis-chatbot:latest'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline execution failed!'
        }
    }
}