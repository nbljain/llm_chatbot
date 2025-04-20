# SQL Chatbot CI/CD Pipeline Documentation

This document provides an overview and explanation of the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the SQL Chatbot project.

## Overview

The SQL Chatbot project uses a comprehensive CI/CD pipeline that automates the process of testing, building, and deploying the application to both staging and production environments. The pipeline is implemented using GitHub Actions and deploys to Azure Kubernetes Service (AKS).

## Pipeline Stages

The CI/CD pipeline consists of the following stages:

### 1. Lint
- **Purpose**: Ensures code quality and consistency
- **Tools**: flake8, black, isort
- **Triggers**: Runs on every push to main branch and on pull requests
- **Actions**:
  - Checks code formatting with black
  - Checks import ordering with isort
  - Checks code style with flake8

### 2. Security Scan
- **Purpose**: Identifies security vulnerabilities in the code and dependencies
- **Tools**: safety, bandit
- **Triggers**: Runs on every push to main branch and on pull requests
- **Actions**:
  - Checks dependencies for known vulnerabilities using safety
  - Performs static code analysis for security issues using bandit
  - Generates and uploads security reports as artifacts

### 3. Unit Tests
- **Purpose**: Verifies individual components function correctly
- **Tools**: pytest, coverage
- **Triggers**: Runs after the Lint stage passes
- **Actions**:
  - Runs unit tests located in tests/unit
  - Generates code coverage reports
  - Uploads coverage data to Codecov

### 4. Integration Tests
- **Purpose**: Verifies components work together correctly
- **Tools**: pytest, coverage
- **Triggers**: Runs after the Unit Tests stage passes
- **Actions**:
  - Runs integration tests located in tests/integration
  - Generates code coverage reports
  - Uploads coverage data to Codecov

### 5. Performance Tests
- **Purpose**: Ensures the application meets performance requirements
- **Tools**: locust
- **Triggers**: Runs after the Integration Tests stage passes
- **Actions**:
  - Starts the application in the background
  - Runs performance tests using locust
  - Generates and uploads performance reports as artifacts

### 6. Build Docker Image
- **Purpose**: Creates a Docker image for deployment
- **Tools**: Docker, Azure Container Registry
- **Triggers**: Runs after all test stages pass, only on pushes to main branch
- **Actions**:
  - Builds a Docker image of the application
  - Pushes the image to Azure Container Registry
  - Tags the image with both latest and commit SHA

### 7. Deploy to Staging
- **Purpose**: Deploys the application to the staging environment
- **Tools**: kubectl, Azure Kubernetes Service
- **Triggers**: Runs after the Build Docker Image stage, only on pushes to main branch
- **Actions**:
  - Authenticates with Azure and AKS
  - Applies Kubernetes manifests from k8s/staging
  - Waits for deployment to complete

### 8. Deploy to Production
- **Purpose**: Deploys the application to the production environment
- **Tools**: kubectl, Azure Kubernetes Service
- **Triggers**: Runs after the Deploy to Staging stage, only on pushes to main branch
- **Actions**:
  - Requires approval from the production environment
  - Authenticates with Azure and AKS
  - Applies Kubernetes manifests from k8s/production
  - Waits for deployment to complete

## Deployment Environments

### Staging
- **Purpose**: Testing environment for final validation before production
- **Scale**: Single replica for cost-efficiency
- **Resources**: Limited CPU and memory
- **Access**: Internal team access only
- **Kubernetes Manifests**: Located in k8s/staging

### Production
- **Purpose**: Live environment for end users
- **Scale**: Multiple replicas with auto-scaling
- **Resources**: Higher CPU and memory allocations
- **Access**: Public access with proper security
- **Kubernetes Manifests**: Located in k8s/production
- **Additional Features**: TLS encryption, persistent storage, auto-scaling

## Required Secrets

The following secrets must be configured in the GitHub repository settings:

1. **AZURE_CREDENTIALS**: JSON credentials for Azure authentication
2. **ACR_LOGIN_SERVER**: Azure Container Registry login server URL
3. **ACR_USERNAME**: Azure Container Registry username
4. **ACR_PASSWORD**: Azure Container Registry password
5. **AKS_RESOURCE_GROUP**: Azure resource group for AKS
6. **AKS_CLUSTER_NAME**: Name of the AKS cluster
7. **OPENAI_API_KEY**: OpenAI API key for NLP functionality

## Workflow Configuration

The pipeline is configured in `.github/workflows/ci-cd.yml`. This file defines all the jobs, steps, and dependencies of the CI/CD process.

## Triggering the Pipeline

The pipeline is triggered in the following ways:
- **Automatically**: On push to the main branch
- **Automatically**: On creation of a pull request to the main branch
- **Manually**: Using the workflow_dispatch event in GitHub Actions

## Monitoring and Troubleshooting

- **Pipeline Status**: Visible in the GitHub Actions tab of the repository
- **Logs**: Available for each job in the GitHub Actions interface
- **Artifacts**: Test reports, coverage data, and other artifacts are uploaded and available for download
- **Deployment Status**: Check the AKS dashboard or use kubectl to view deployment status

## Best Practices

1. **Pull Requests**: Always use pull requests for changes to the main branch
2. **Tests**: Ensure all tests pass locally before pushing changes
3. **Secrets**: Never commit secrets or credentials to the repository
4. **Dependencies**: Regularly update dependencies to address security vulnerabilities
5. **Documentation**: Update this documentation when making changes to the CI/CD process

## Future Improvements

- Add blue/green deployment strategy
- Implement canary releases for production
- Add smoke tests after deployment
- Integrate with a monitoring solution for post-deployment verification
- Add notifications for deployment status (Slack, email, etc.)