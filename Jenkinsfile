pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "data-analysis-chatbot:latest"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                bat 'python -m pip install --upgrade pip'
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                bat 'echo "Tests would go here"'
                // Add actual tests like:
                // sh 'python -m pytest'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker-compose build'
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    // Push Docker Image to Docker Hub (or your container registry)
                    bat 'docker tag data-analysis-chatbot:latest arindamcse/data-analysis-chatbot:latest'
                    bat 'docker push arindamcse/data-analysis-chatbot:latest'
                }
            }
        }

        stage('Deploy with Docker Compose') {
            steps {
                bat 'docker-compose up -d'
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
