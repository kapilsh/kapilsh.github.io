---
title: System Design Interview Process
description: >-
  How to approach system design interview
date: 2021-09-23
categories: [Blog, Guide]
tags: [Interview, System Design]
pin: false
author: ks
---

If you have gone through any technical interview process in the recent past, chances are that you have come across at least one round of system design interview. There are various great resources out there to prepare for these interviews, but one of the important aspect is knowing how to manage time while talking about a system. In this post, I will outline a process that I followed in my recent technical interview processes. I am writing this post with the assumption of a 45 min interview.

## Requirements (5 min)
Talk about details of different requirements and ask questions
- Functional
- Non-functional
- Extensions

### Common Functional Requirements

- What will users use the product for?
- How will users interact with the system?
- How many users? Scale?
- User Patterns
  - How many photos/videos, etc.?
  - How many posts a day?
  - Average sizes?
- Other product details

### Common Non-functional Requirements
- Availability
- Reliability
- Consistency
- Latency

### Extensions
- Push notifications
- Analytics
- Data/ML
- REST/WEBSOCKET API

## Capacity/Estimations (Back of the Envelope Math) (5 min)
Estimate what the requirements will mean in terms of storage and infrastructure costs
- Read heavy/write heavy?
- What is the scale?
  - How many tweets/users?
  - How many connections edges/degrees of separations?
  - How much storage?
  - Over how many years?
  - Storage types - text, photos, videos, etc.
  - Bandwidth / Latency
  - Memory

## System Interface Design (5 min)
- General CRUD system (Create, Read, Update, Delete)
- How do users interact with the system and give it an interface
- System interface should also help with database design. Typically, parameters to the function calls should be in database in one form or the other.

### REST API Design
Functions or GET/POST requests
- getTweets(), postTweet(), deleteTweets(), generateTimeline(), getTopKTweets()
- getNearestDrivers(), getNearestFriends()
- getOfflineMessages()

### WEBSOCKET API Design
Determine the datatypes that need to be streamed - protocol can be json, binary

### Other Considerations
- Data fetching should keep into account the screen size i.e., grab lower resolution pictures for smaller screens such as mobile phones
- Tradeoffs around how users can interact with the system. For example, REST vs Long Polling vs Websockets
- Video codecs, resolutions
- Photo resolution and pixel sizes


## Data Model (5 min)
- Get all the entities that are the part of the system
  - User / Tweet / Post / UserFollow / Friends
  - Driver / User
  - URL / Shortened URL
  - PictureMetadata/ VideoMetadata
- Talk about types of storage
- SQL vs NoSQL
- ACID
- Object storage for pictures, videos, etc.
- Sharding schemes

### Common Considerations
- Tradeoffs between database types
- Tradeoffs between sharding schemes
- Is ACID needed?

## System Block Diagram (5 min)
- Draw all the key components and how they interact with each other
- Database / File storage
- Load balancers
- Caching
- Horizontal scaling
- Replication
- Service coordination / Registry
- Health check / monitoring

### Key Ideas:
- Clients connect to app server or load balancers connected app servers
- Add load balancers between fast and slow components or if there is mismatch between number of entities between two layers. For example, millions of users connect to a few hundred servers => need a load balancer
- Add load balancer between DB and app server
- Add Cache between DB and services connecting to DB
- Follow 80-20 rule in caching
- Add fast-lanes for hot topics/ users
- Job Queues for calculation tasks such as number of likes etc.
- Two types of services - Storage and Compute


## Detailed Design (10 min)
This is the most important part of the interview where you can demonstrate your knowledge and skills:
- Discuss the high-level design and pick the important components
- Discuss the top 2-3 components and go deeper into the details
  - Different approaches and pros/cons of each
  - Tradeoffs, Tradeoffs, Tradeoffs (IMPORTANT!)
  - How to handle hot users
  - How to optimize for quick fetch of data i.e. where should caching go
  - Load balancers - round-robin vs load-centric
  - Retry mechanisms - exponential decay

## Fault Tolerance and Bottlenecks (2-3 min)
What could go wrong and how to resolve issues?
- Find the Single Points of Failures and mitigate
- Replication - what if we lose a bunch of servers
- How do users/services reconnect
- Monitoring the performance - alerts, etc
- Metrics, tracing, observability

Hope this helps other people in approaching their system design interviews systematically!
