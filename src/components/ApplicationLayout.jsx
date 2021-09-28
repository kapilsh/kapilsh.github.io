import React from "react";

import {Layout, Space, PageHeader, Tag, Card} from "antd";

import Sidebar from "./Sidebar";
import Content from "./Content";
import posts from "../posts";
import {Link} from "react-router-dom";

const {Sider} = Layout;

const PostTag = ({text}) => (
    <Tag style={{"marginTop": 5, "marginBottom": 5}} color={"#1890ff"}>
        <Space/>
        <span>#{text.split(" ").join("").toLowerCase()}</span>
        <Space/>
    </Tag>
);

class AppLayout extends React.Component {
    constructor(props) {
        super(props);
        this.state = {collapsed: false};
    }

    render() {
        const postsCopy = [...posts]
        return (
            <Layout style={{minHeight: "100vh"}}>
                <Sidebar/>
                <Content collapsed={this.state.collapsed}/>
                <Sider
                    width={300}
                    breakpoint="lg"
                    collapsedWidth={0}
                    onCollapse={(collapsed, type) => {
                        this.setState({collapsed: collapsed});
                    }}
                    style={{
                        overflow: "auto",
                        height: "100vh",
                        position: "fixed",
                        right: 0,
                    }}
                >
                    <PageHeader className="site-page-header" title="More Posts"/>
                    {postsCopy.sort(function (p1, p2) {
                        const key1 = new Date(p1.date);
                        const key2 = new Date(p2.date);
                        if (key1 < key2) return -1;
                        if (key1 > key2) return 1;
                        return 0;
                    }).map((post, index) => {
                        const link = post.title.split(" ").join("-").toLowerCase();
                        return (
                            <Link to={`/posts/${link}`}>
                                <Card
                                    hoverable
                                    size={"small"}
                                    style={{width: "100%"}}
                                    key={index}
                                >
                                    <p>{post.title}</p>
                                    <p>
                                        {post.tags.map((t, i) => {
                                            return <PostTag text={t} key={i}/>;
                                        })}
                                    </p>
                                </Card>
                            </Link>
                        );
                    })}
                </Sider>
            </Layout>
        );
    }
}

export default AppLayout;
