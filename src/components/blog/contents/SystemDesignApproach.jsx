import React from "react";
import {Typography} from "antd";

const {Title, Paragraph} = Typography;

class SystemDesignApproach extends React.Component {
    render() {
        return (<div>
            <Typography>
                <Paragraph>
                    If you have gone through any technical interview process in the recent past, chances are that you
                    have come across at least one round of system design interview. There are various great resources
                    out there to prepare for these interviews, but one of the important
                    aspect is knowing how to manage time while talking about a system. In this post, I will
                    outline a process that I followed in my recent technical interview processes. I am writing this post
                    with the assumption of a 45 min interview.
                </Paragraph>
                <Title level={3}>
                    Requirements (5 min)
                </Title>
                <Paragraph>
                    Talk about details of different requirements and ask questions
                    <ul>
                        <li>
                            Functional
                        </li>
                        <li>
                            Non-functional
                        </li>
                        <li>
                            Extensions
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    Common Functional Requirements
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            What will users use the product for?
                        </li>
                        <li>
                            How will users interact with the system?
                        </li>
                        <li>
                            How many users? Scale?
                        </li>
                        <li>
                            User Patterns
                            <ul>
                                <li>
                                    How many photos/videos, etc.?
                                </li>
                                <li>
                                    How many posts a day?
                                </li>
                                <li>
                                    Average sizes?
                                </li>
                            </ul>
                        </li>
                        <li>
                            Other product details
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    Common Non-functional Requirements
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Availability
                        </li>
                        <li>
                            Reliability
                        </li>
                        <li>
                            Consistency
                        </li>
                        <li>
                            Latency
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    Extensions
                </Title>
                <ul>
                    <li>
                        Push notifications
                    </li>
                    <li>
                        Analytics
                    </li>
                    <li>
                        Data/ML
                    </li>
                    <li>
                        REST/WEBSOCKET API
                    </li>
                </ul>
                <Title level={3}>
                    Capacity/Estimations (Back of the Envelope Math) (5 min)
                </Title>
                <Paragraph>
                    Estimate what the requirements will mean in terms of storage and infrastructure costs
                    <ul>
                        <li>
                            Read heavy/write heavy?
                        </li>
                        <li>
                            What is the scale?
                            <ul>
                                <li>
                                    How many tweets/users?
                                </li>
                                <li>
                                    How many connections edges/degrees of separations?
                                </li>
                                <li>
                                    How much storage?
                                </li>
                                <li>
                                    Over how many years?
                                </li>
                                <li>
                                    Storage types - text, photos, videos, etc.
                                </li>
                                <li>
                                    Bandwidth / Latency
                                </li>
                                <li>
                                    Memory
                                </li>
                            </ul>
                        </li>
                    </ul>
                </Paragraph>
                <Title level={3}>
                    System Interface Design (5 min)
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            General CRUD system (Create, Read, Update, Delete)
                        </li>
                        <li>
                            How do users interact with the system and give it an interface
                        </li>
                        <li>
                            System interface should also help with database design. Typically, parameters to the
                            function calls should be in database in one form or the other.
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    REST API Design
                </Title>
                <Paragraph>
                    Functions or GET/POST requests
                    <ul>
                        <li>
                            getTweets(), postTweet(), deleteTweets(), generateTimeline(), getTopKTweets()
                        </li>
                        <li>
                            getNearestDrivers(), getNearestFriends()
                        </li>
                        <li>
                            getOfflineMessages()
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    WEBSOCKET API Design
                </Title>
                <Paragraph>
                    Determine the datatypes that need to be streamed - protocol can be json, binary
                </Paragraph>
                <Title level={4}>
                    Other Considerations
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Data fetching should keep into account the screen size i.e., grab lower resolution pictures
                            for smaller screens such as mobile phones
                        </li>
                        <li>
                            Tradeoffs around how users can interact with the system. For example, REST vs Long Polling
                            vs
                            Websockets
                        </li>
                        <li>
                            Video codecs, resolutions
                        </li>
                        <li>
                            Photo resolution and pixel sizes
                        </li>
                    </ul>
                </Paragraph>
                <Title level={3}>
                    Data Model (5 min)
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Get all the entities that are the part of the system
                            <ul>
                                <li>
                                    User / Tweet / Post / UserFollow / Friends
                                </li>
                                <li>
                                    Driver / User
                                </li>
                                <li>
                                    URL / Shortened URL
                                </li>
                                <li>
                                    PictureMetadata/ VideoMetadata
                                </li>
                            </ul>
                        </li>
                        <li>
                            Talk about types of storage
                        </li>
                        <li>
                            SQL vs NoSQL
                        </li>
                        <li>
                            ACID
                        </li>
                        <li>
                            Object storage for pictures, videos, etc.
                        </li>
                        <li>
                            Sharding schemes
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    Common Considerations
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Tradeoffs between database types
                        </li>
                        <li>
                            Tradeoffs between sharding schemes
                        </li>
                        <li>
                            Is ACID needed?
                        </li>
                    </ul>
                </Paragraph>
                <Title level={3}>
                    System Block Diagram (5 min)
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Draw all the key components and how they interact with each other
                        </li>
                        <li>
                            Database / File storage
                        </li>
                        <li>
                            Load balancers
                        </li>
                        <li>
                            Caching
                        </li>
                        <li>
                            Horizontal scaling
                        </li>
                        <li>
                            Replication
                        </li>
                        <li>
                            Service coordination / Registry
                        </li>
                        <li>
                            Health check / monitoring
                        </li>
                    </ul>
                </Paragraph>
                <Title level={4}>
                    Key Ideas:
                </Title>
                <Paragraph>
                    <ul>
                        <li>
                            Clients connect to app server or load balancers connected app servers
                        </li>
                        <li>
                            Add load balancers between fast and slow components or if there is mismatch between number
                            of entities between two layers. For example, millions of users connect to a few hundred
                            servers => need a load balancer
                        </li>
                        <li>
                            Add load balancer between DB and app server
                        </li>
                        <li>
                            Add Cache between DB and services connecting to DB
                        </li>
                        <li>
                            Follow 80-20 rule in caching
                        </li>
                        <li>
                            Add fast-lanes for hot topics/ users
                        </li>
                        <li>
                            Job Queues for calculation tasks such as number of likes etc.
                        </li>
                        <li>
                            Two types of services - Storage and Compute
                        </li>
                    </ul>
                </Paragraph>
                <Title level={3}>
                    Detailed Design (10 min)
                </Title>
                <Paragraph>
                    This is the most important part of the interview where you can demonstrate your knowledge and
                    skills:
                    <ul>
                        <li>
                            Discuss the high-level design and pick the important components
                        </li>
                        <li>
                            Discuss the top 2-3 components and go deeper into the details
                            <ul>
                                <li>
                                    Different approaches and pros/cons of each
                                </li>
                                <li>
                                    Tradeoffs, Tradeoffs, Tradeoffs (IMPORTANT!)
                                </li>
                                <li>
                                    How to handle hot users
                                </li>
                                <li>
                                    How to optimize for quick fetch of data i.e. where should caching go
                                </li>
                                <li>
                                    Load balancers - round-robin vs load-centric
                                </li>
                                <li>
                                    Retry mechanisms - exponential decay
                                </li>
                            </ul>
                        </li>
                    </ul>
                </Paragraph>
                <Title level={3}>
                    Fault Tolerance and Bottlenecks (2-3 min)
                </Title>
                <Paragraph>
                    What could go wrong and how to resolve issues?
                    <ul>
                        <li>
                            Find the Single Points of Failures and mitigate
                        </li>
                        <li>
                            Replication - what if we lose a bunch of servers
                        </li>
                        <li>
                            How do users/services reconnect
                        </li>
                        <li>
                            Monitoring the performance - alerts, etc
                        </li>
                        <li>
                            Metrics, tracing, observability
                        </li>
                    </ul>
                </Paragraph>
                <br/>
                Hope this helps other people in approaching their system design interviews systematically!
            </Typography>
        </div>)
    }
}

export default SystemDesignApproach;