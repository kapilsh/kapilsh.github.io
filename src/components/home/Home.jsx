import React from "react";
import { Card, Row, Col, Tag, Space, Pagination, Typography } from "antd";
import posts from "../../posts";

import { TagOutlined, CalendarOutlined } from "@ant-design/icons";
import { Link } from "react-router-dom";

const { Title, Paragraph } = Typography;

const PostTag = ({ text }) => (
  <Tag style={{"marginTop": 5, "marginBottom": 5}} color={"#1890ff"}>
    <Space />
    <TagOutlined />
    <span>{text}</span>
    <Space />
  </Tag>
);

const BlogDate = ({ date }) => (
  <Space>
    <CalendarOutlined />
    {date}
  </Space>
);

const numEachPage = 6;

class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      minValue: 0,
      maxValue: numEachPage,
    };
  }

  render() {
    return (
      <Row>
        <Col span={24}>
          {posts
            .slice(this.state.minValue, this.state.maxValue)
            .map((post, index) => {
              const link = post.title.split(" ").join("-").toLowerCase();
              return (
                <div key={index}>
                  <Link to={`/posts/${link}`}>
                    <Card
                      hoverable
                      style={{ width: "100%" }}
                      cover={
                        <img
                          alt={post.image}
                          src={post.image}
                          style={{
                            width: "100%",
                            height: "200px",
                            objectFit: "cover",
                          }}
                        />
                      }
                    >
                      <Typography>
                        <br />
                        <Title level={2}>{post.title}</Title>
                        <Title level={4}>
                          {post.description}{" "}
                          <Tag>
                            <BlogDate date={post.date} />
                          </Tag>
                        </Title>
                      </Typography>
                      {post.tags.map((t, i) => {
                        return <PostTag text={t} key={i} />;
                      })}
                    </Card>
                    <br />
                  </Link>
                </div>
              );
            })}
          <Pagination
            defaultCurrent={1}
            defaultPageSize={numEachPage}
            onChange={(value) => {
              this.setState({
                minValue: (value - 1) * numEachPage,
                maxValue: value * numEachPage,
              });
            }}
            total={posts.length}
          />
        </Col>
      </Row>
    );
  }
}

export default Home;
